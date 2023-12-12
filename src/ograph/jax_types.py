from typing import Protocol

from jaxtyping import ArrayLike, Float, Int

Arr = ArrayLike

EdgeFeat = Float[Arr, "edge_dim"]
NodeFeat = Float[Arr, "node_dim"]
NodeIdx = Int[Arr, ""]

EEdgeFeat = Float[Arr, "num_edges edge_dim"]
ENodeFeat = Float[Arr, "num_edges node_dim"]
NNodeFeat = Float[Arr, "num_nodes node_dim"]

NNEdgeFeat = Float[Arr, "num_recv_nodes num_send_nodes node_dim"]

ENodeIdx = Int[Arr, "num_edges"]

EFloat = Float[Arr, "num_edges"]

class MessageFn(Protocol):
    def __call__(self, edge: EdgeFeat, sender: NodeFeat, receiver: NodeFeat) -> EdgeFeat:
        ...


class UpdateFn(Protocol):
    def __call__(self, n_node: NNodeFeat, n_aggr_msg: NNodeFeat) -> NNodeFeat:
        ...


class Aggregator(Protocol):
    def sparse(self, e_edge: EEdgeFeat, e_recv_idx: ENodeIdx, n_nodes: int) -> NNodeFeat:
        """Aggregation function for general graphs."""
        ...

    def dense(self, nn_edge: NNEdgeFeat) -> NNodeFeat:
        """Aggregation function for dense graphs."""
        ...
