from ograph.graph_types import Graph
import pytest
import numpy as np
from og.tree_utils import tree_stack


def test_graph_static():
    g1 = Graph(1, np.array(2), np.array(3), np.array(4), np.array(5), np.array(6))
    g2 = Graph(1, np.array(12), np.array(13), np.array(14), np.array(15), np.array(16))

    b_graph = tree_stack([g1, g2], axis=0)
    assert b_graph.num_nodes == 1

    # We can only stack if the num_nodes is the same.
    g3 = Graph(2, np.array(12), np.array(13), np.array(14), np.array(15), np.array(16))
    with pytest.raises(ValueError) as e_info:
        b_graph = tree_stack([g1, g3], axis=0)


if __name__ == "__main__":
    test_graph_static()
