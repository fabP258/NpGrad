from npgrad.node import Node
import numpy as np


def test_softmax_forward1d():
    """Test softmax with 1D array"""

    # vector
    x = np.array([-2.25066624, -1.7278112, 0.57369723, 2.18999384, -1.92979703])
    y_expected = np.array([0.0094566, 0.01595173, 0.15934568, 0.80221172, 0.01303427])
    out_node = Node(x)
    y = out_node.softmax()

    assert x.shape == y.value.shape, "Input and output of softmax shall have same shape"
    assert np.allclose(y.value, y_expected)


def test_softmax_forward2d():
    """Test softmax with 2D array"""

    # vector as matrix
    x = np.array([[-2.25066624, -1.7278112, 0.57369723, 2.18999384, -1.92979703]])
    y_expected = np.array([[0.0094566, 0.01595173, 0.15934568, 0.80221172, 0.01303427]])
    out_node = Node(x)
    y = out_node.softmax()

    assert x.shape == y.value.shape, "Input and output of softmax shall have same shape"
    assert np.allclose(y.value, y_expected)

    # matrix
    x = np.array(
        [
            [0.5837299, 1.16565558, 1.07832413, -1.35152097, 1.59659022],
            [-0.54887338, -2.3345048, 0.93418145, 0.45424355, 0.44316959],
            [-0.198241, -2.4081991, -1.81922612, 1.56015527, -1.38002645],
        ]
    )
    y_expected = np.array(
        [
            [0.13647839, 0.2442255, 0.22380173, 0.01970585, 0.37578854],
            [0.09092907, 0.015248, 0.40066858, 0.24794246, 0.2452119],
            [0.13482062, 0.01479053, 0.02665455, 0.78238068, 0.04135361],
        ]
    )
    out_node = Node(x)
    y = out_node.softmax()

    assert x.shape == y.value.shape, "Input and output of softmax shall have same shape"
    assert np.allclose(y.value, y_expected)
