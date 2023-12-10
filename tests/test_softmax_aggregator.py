import ipdb
import numpy as np

from ograph.aggregators import SoftmaxAggregator
from ograph.jax_types import EdgeFeat


def test_softmax_aggregator():
    def avg_gate_feat(edge_feat: EdgeFeat) -> EdgeFeat:
        return np.array(1.0)

    def sum_gate_feat(edge_feat: EdgeFeat) -> EdgeFeat:
        return edge_feat.sum()

    e_edgefeat = np.array([[1], [2], [3]])
    e_recvidx = np.array([0, 0, 1])
    num_nodes = 2

    #############################################################
    aggregator = SoftmaxAggregator(avg_gate_feat)
    n_nodefeat = aggregator.sparse(e_edgefeat, e_recvidx, num_nodes)
    assert n_nodefeat.shape == (2, 1)

    n_nodefeat_true = np.array([1.5, 3])[:, None]
    np.testing.assert_allclose(n_nodefeat, n_nodefeat_true)
    #############################################################
    aggregator = SoftmaxAggregator(sum_gate_feat)
    n_nodefeat = aggregator.sparse(e_edgefeat, e_recvidx, num_nodes)
    assert n_nodefeat.shape == (2, 1)

    e1, e2 = np.exp(1), np.exp(2)
    n_nodefeat_true = np.array([(e1 * 1 + e2 * 2) / (e1 + e2), 3])[:, None]
    np.testing.assert_allclose(n_nodefeat, n_nodefeat_true)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        test_softmax_aggregator()
