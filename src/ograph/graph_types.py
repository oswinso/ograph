from typing import Generic, NamedTuple, TypeVar, Union, get_type_hints

import jax.tree_util as jtu
from jax._src.tree_util import GetAttrKey
from jaxtyping import Int
from og.none import get_or

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

    def __new__(
        cls,
        num_nodes: int,
        num_edges: Int[Arr, "n_graph"],
        n_nodefeat: NNodeFeat,
        e_edgefeat: EEdgeFeat,
        e_sendidx: ENodeIdx,
        e_recvidx: ENodeIdx,
    ) -> "Graph":
        assert isinstance(num_nodes, int), "num_nodes must be a (concrete) int!"
        tup = (num_nodes, num_edges, n_nodefeat, e_edgefeat, e_sendidx, e_recvidx)
        self = tuple.__new__(cls, tup)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.n_nodefeat = n_nodefeat
        self.e_edgefeat = e_edgefeat
        self.e_sendidx = e_sendidx
        self.e_recvidx = e_recvidx
        return self

    def replace(
        self,
        num_nodes: int = None,
        num_edges: Int[Arr, "n_graph"] = None,
        n_nodefeat: NNodeFeat = None,
        e_edgefeat: EEdgeFeat = None,
        e_sendidx: ENodeIdx = None,
        e_recvidx: ENodeIdx = None,
    ) -> "Graph":
        return Graph(
            num_nodes=get_or(num_nodes, self.num_nodes),
            num_edges=get_or(num_edges, self.num_edges),
            n_nodefeat=get_or(n_nodefeat, self.n_nodefeat),
            e_edgefeat=get_or(e_edgefeat, self.e_edgefeat),
            e_sendidx=get_or(e_sendidx, self.e_sendidx),
            e_recvidx=get_or(e_recvidx, self.e_recvidx),
        )

    def _replace(self, **kwargs):
        return self.replace(**kwargs)


class ComplGraph(NamedTuple):
    """Complete graph."""

    # (n_nodes, node_dim)
    n_nodefeat: NNodeFeat
    # (n_recv, n_send, node_dim)
    nn_edgefeat: NNEdgeFeat

    @property
    def batch_shape(self):
        return self.nn_edgefeat.shape[:-3]

    @property
    def is_single(self) -> bool:
        return self.batch_shape == tuple()

    def replace(self, n_nodefeat: NNodeFeat = None, nn_edgefeat: NNEdgeFeat = None) -> "ComplGraph":
        return ComplGraph(
            n_nodefeat=get_or(n_nodefeat, self.n_nodefeat),
            nn_edgefeat=get_or(nn_edgefeat, self.nn_edgefeat),
        )


AnyGraph = Union[Graph, ComplGraph]
