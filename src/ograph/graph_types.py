from typing import NamedTuple, Union

from jaxtyping import Int

from ograph.jax_types import Arr, EEdgeFeat, ENodeIdx, NNEdgeFeat, NNodeFeat


class Graph(NamedTuple):
    num_nodes: Int[Arr, "n_graph"]
    num_edges: Int[Arr, "n_graph"]
    n_nodefeat: NNodeFeat
    e_edgefeat: EEdgeFeat
    e_sendidx: ENodeIdx
    e_recvidx: ENodeIdx

    @property
    def batch_shape(self):
        return self.num_nodes.shape

    @property
    def is_single(self) -> bool:
        return self.batch_shape == tuple()

    @property
    def n_graphs(self) -> int:
        if self.is_single:
            return 1
        assert len(self.num_nodes) == len(self.num_edges)
        return len(self.num_nodes)


class ComplGraph(NamedTuple):
    """Complete graph."""

    num_nodes: Int[Arr, "n_graph"]
    n_nodefeat: NNodeFeat
    # (n_recv, n_send, node_dim)
    nn_edgefeat: NNEdgeFeat

    @property
    def batch_shape(self):
        return self.num_nodes.shape

    @property
    def is_single(self) -> bool:
        return self.batch_shape == tuple()

    @property
    def n_graphs(self) -> int:
        if self.is_single:
            return 1
        return len(self.num_nodes)


AnyGraph = Union[Graph, ComplGraph]
