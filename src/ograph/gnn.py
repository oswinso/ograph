import functools as ft
from typing import NamedTuple, Protocol, Type, TypeVar

import einops as ei
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ograph.aggregators import SoftmaxAggregator
from ograph.graph_types import AnyGraph, ComplGraph, Graph
from ograph.jax_types import Aggregator, EdgeFeat, EEdgeFeat, MessageFn, NNodeFeat, NodeFeat, UpdateFn
from ograph.jax_utils import safe_get

_AnyGraph = TypeVar("_AnyGraph", Graph, ComplGraph)


class GNNUpdate(NamedTuple):
    message: MessageFn
    aggregate: Aggregator
    update: UpdateFn

    def __call__(self, graph: _AnyGraph) -> _AnyGraph:
        if isinstance(graph, Graph):
            return self.run(graph)
        if isinstance(graph, ComplGraph):
            return self.run_compl(graph)

        raise ValueError(f"Unknown graph type {type(graph)}")

    def run(self, graph: Graph) -> Graph:
        assert graph.is_single

        e_nodefeat_send = jtu.tree_map(lambda n: safe_get(n, graph.e_sendidx), graph.n_nodefeat)
        e_nodefeat_recv = jtu.tree_map(lambda n: safe_get(n, graph.e_recvidx), graph.n_nodefeat)

        # message passing
        e_edgefeat = jax.vmap(self.message)(graph.e_edgefeat, e_nodefeat_send, e_nodefeat_recv)

        # aggregate messages
        n_aggr_msg = self.aggregate.sparse(e_edgefeat, graph.e_recvidx, graph.num_nodes)

        # update nodes
        n_nodefeat = self.update(graph.n_nodefeat, n_aggr_msg)

        return graph._replace(n_nodefeat=n_nodefeat, e_edgefeat=e_edgefeat)

    def run_compl(self, graph: ComplGraph) -> ComplGraph:
        assert graph.is_single

        # (n_recv, n_send, node_dim)
        nn_edgefeat = graph.nn_edgefeat
        # (1, n_send, node_dim)
        In_send = graph.n_nodefeat[None, :, :]
        # (n_recv, 1, node_dim
        nI_recv = graph.n_nodefeat[:, None, :]

        # message passing.
        vmap_message = jax.vmap(jax.vmap(self.message, in_axes=(0, 0, None)), in_axes=(0, None, 0))
        nn_edgefeat = vmap_message(nn_edgefeat, In_send[0, :, :], nI_recv[:, 0, :])

        # aggregate messages
        n_aggr_msg = self.aggregate.dense(nn_edgefeat)

        # update nodes
        n_nodefeat = self.update(graph.n_nodefeat, n_aggr_msg)

        return graph._replace(n_nodefeat=n_nodefeat, nn_edgefeat=nn_edgefeat)


class SoftmaxGNN(nn.Module):
    msg_net_cls: Type[nn.Module]
    gate_feat_cls: Type[nn.Module]
    update_net_cls: Type[nn.Module]
    msg_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, graph: AnyGraph) -> AnyGraph:
        kernel_init = nn.initializers.xavier_uniform

        def message(edge: EdgeFeat, sender: NodeFeat, receiver: NodeFeat) -> EdgeFeat:
            feats = jnp.concatenate([edge, sender, receiver], axis=-1)
            feats = self.msg_net_cls()(feats)
            feats = nn.Dense(self.msg_dim, kernel_init=kernel_init())(feats)
            return feats

        def update(n_node: NNodeFeat, n_aggr_msg: NNodeFeat) -> NNodeFeat:
            feats = jnp.concatenate([n_node, n_aggr_msg], axis=-1)
            feats = self.update_net_cls()(feats)
            feats = nn.Dense(self.out_dim, kernel_init=kernel_init())(feats)
            return feats

        def get_gate_feats(e_edge: EEdgeFeat) -> EEdgeFeat:
            e_gate_feats = self.gate_feat_cls()(e_edge)
            e_gate_feats = nn.Dense(1, kernel_init=kernel_init())(e_gate_feats).squeeze(-1)
            return e_gate_feats

        aggregator = SoftmaxAggregator(get_gate_feats)
        update_fn = GNNUpdate(message, aggregator, update)
        return update_fn(graph)
