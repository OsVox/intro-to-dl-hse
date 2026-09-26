import numpy as np

from modules import LowRankLinear, SGD


def test_low_rank_linear():
    rng = np.random.default_rng(9)
    for bias in (True, False):
        layer = LowRankLinear(4, 3, max_rank=2, bias=bias)
        x = rng.normal(size=(2, 4))
        upstream = rng.normal(size=(2, 3))

        expected = x @ layer.weight.T
        if bias:
            expected += layer.bias
        assert np.allclose(layer(x), expected)
        assert np.linalg.matrix_rank(layer.weight) <= 2

        grad_input = layer.backward(x, upstream)
        parameter_grads = [grad.copy() for grad in layer.parameters_grad()]
        eps = 1e-6

        for index in np.ndindex(x.shape):
            x[index] += eps
            plus = np.sum(layer(x) * upstream)
            x[index] -= 2 * eps
            minus = np.sum(layer(x) * upstream)
            x[index] += eps
            assert np.isclose(grad_input[index], (plus - minus) / (2 * eps), atol=1e-7)

        for parameter, gradient in zip(layer.parameters(), parameter_grads):
            for index in np.ndindex(parameter.shape):
                parameter[index] += eps
                plus = np.sum(layer(x) * upstream)
                parameter[index] -= 2 * eps
                minus = np.sum(layer(x) * upstream)
                parameter[index] += eps
                assert np.isclose(gradient[index], (plus - minus) / (2 * eps), atol=1e-7)

        layer(x)
        layer.backward(x, upstream)
        for gradient, first in zip(layer.parameters_grad(), parameter_grads):
            assert np.allclose(gradient, 2 * first)

        SGD(layer, lr=0.1).step()
        assert np.linalg.matrix_rank(layer.weight) <= 2
        layer.zero_grad()
        assert all(np.count_nonzero(gradient) == 0 for gradient in layer.parameters_grad())

    assert LowRankLinear(4, 3).rank == 3
    print('test_low_rank_linear ... OK')
