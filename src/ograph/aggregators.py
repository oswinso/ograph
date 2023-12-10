from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ograph.jax_types import Aggregator, EdgeFeat, EEdgeFeat, EFloat, ENodeIdx, NNEdgeFeat, NNodeFeat


def segment_softmax(
    e_logits: EFloat,
    e_recvidx: ENodeIdx,
    num_nodes: int,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> EFloat:
    # Subtract max for numerical stability.
    n_logit_max = jax.ops.segment_max(e_logits, e_recvidx, num_nodes, indices_are_sorted, unique_indices)
    e_logits = e_logits - n_logit_max[e_recvidx]
    e_odds = jnp.exp(e_logits)
    # Compute the normalizer.
    n_sumodds = jax.ops.segment_sum(e_odds, e_recvidx, num_nodes, indices_are_sorted, unique_indices)
    e_sumodds = n_sumodds[e_recvidx]
    e_softmax = e_odds / e_sumodds
    assert e_softmax.shape == e_logits.shape
    return e_softmax


class SoftmaxAggregator(NamedTuple):
    get_gate_feats: Callable[[EdgeFeat], EdgeFeat]

    def sparse(self, e_edgefeat: EEdgeFeat, e_recvidx: ENodeIdx, num_nodes: int) -> NNodeFeat:
        n_edges = len(e_edgefeat)

        e_gate_feats = jax.vmap(self.get_gate_feats)(e_edgefeat)
        assert e_gate_feats.shape == (n_edges, )

        e_attn_weights = segment_softmax(e_gate_feats, e_recvidx, num_nodes)
        assert e_attn_weights.shape[0] == e_edgefeat.shape[0]

        e_weighted_edgefeat = e_attn_weights[:, None] * e_edgefeat
        n_aggr_msg = jax.ops.segment_sum(e_weighted_edgefeat, e_recvidx, num_nodes)
        assert n_aggr_msg.ndim == 2 and len(n_aggr_msg) == num_nodes
        return n_aggr_msg

    def dense(self, nn_edgefeat: NNEdgeFeat) -> NNodeFeat:
        # nn_edgefeat: (n_recv, n_send, node_dim)
        # (n_recv, n_send)
        nn_gate_feats = jax.vmap(jax.vmap(self.get_gate_feats))(nn_edgefeat)
        # (n_recv, n_send)
        nn_attn_weights = jax.nn.softmax(nn_gate_feats, axis=1)
        nn_weighted_edgefeat = nn_attn_weights[:, :, None] * nn_edgefeat
        # (n_recv, node_dim)
        n_aggr_msg = jnp.sum(nn_weighted_edgefeat, axis=1)
        assert len(n_aggr_msg) == nn_edgefeat.shape[0]
        return n_aggr_msg
