from typing import Generic, NamedTuple, TypeVar, Union, get_type_hints

import jax.tree_util as jtu
from jax._src.tree_util import GetAttrKey
from jaxtyping import Int

from ograph.jax_types import Arr, EEdgeFeat, ENodeIdx, NNEdgeFeat, NNodeFeat


@jtu.register_pytree_with_keys_class
class Graph(tuple):
    # This is static!
    num_nodes: int
    num_edges: Int[Arr, "n_graph"]
    n_nodefeat: NNodeFeat
    e_edgefeat: EEdgeFeat
    e_sendidx: ENodeIdx
    e_recvidx: ENodeIdx

    def __new__(
        cls,
        num_nodes: int,
        num_edges: Int[Arr, "n_graph"],
        n_nodefeat: NNodeFeat,
        e_edgefeat: EEdgeFeat,
        e_sendidx: ENodeIdx,
        e_recvidx: ENodeIdx,
    ) -> "Graph":
        tup = (num_nodes, num_edges, n_nodefeat, e_edgefeat, e_sendidx, e_recvidx)
        self = tuple.__new__(cls, tup)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.n_nodefeat = n_nodefeat
        self.e_edgefeat = e_edgefeat
        self.e_sendidx = e_sendidx
        self.e_recvidx = e_recvidx
        return self

    @property
    def batch_shape(self):
        return self.num_edges.shape

    @property
    def is_single(self) -> bool:
        return self.batch_shape == tuple()

    def tree_flatten_with_keys(self):
        flat_contents = [(GetAttrKey(k), getattr(self, k)) for k in get_type_hints(Graph).keys()]
        # Remove num_nodes from flat_contents, put it in aux_data.
        aux_data = flat_contents.pop(0)[1]
        assert isinstance(aux_data, int)
        return flat_contents, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        assert isinstance(aux_data, int)
        children = [aux_data, *children]
        assert len(children) == 6
        return cls(*children)


class ComplGraph(NamedTuple):
    """Complete graph."""
    # (n_nodes, node_dim)
    n_nodefeat: NNodeFeat
    # (n_recv, n_send, node_dim)
    nn_edgefeat: NNEdgeFeat

    @property
    def batch_shape(self):
        return self.n_nodefeat.shape[:-2]

    @property
    def is_single(self) -> bool:
        return self.batch_shape == tuple()


AnyGraph = Union[Graph, ComplGraph]
