# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from sklearn.utils import check_random_state

from carl.distributions import Histogram


def test_histogram():
    X = np.arange(11).reshape(-1, 1)
    h = Histogram(bins=11)
    h.fit(X)

    assert_array_almost_equal(
        h.pdf([[0.0], [1.0], [10.0], [-0.5], [10.5]]),
        [0.1, 0.1, 0.1, 0., 0.])

    assert_array_almost_equal(
        h.nll([[0.0], [1.0], [10.0], [-0.5], [10.5]]),
        -np.log(h.pdf([[0.0], [1.0], [10.0], [-0.5], [10.5]])))

    X = h.rvs(10000, random_state=1)
    assert np.abs(np.mean(X) - 5.0) < 0.05
    assert X.min() >= 0.0
    assert X.max() <= 10.0


def test_histogram_sample_weight():
    X = np.arange(11).reshape(-1, 1)
    w = np.ones(len(X)) / len(X)

    h1 = Histogram(bins=11)
    h1.fit(X)
    h2 = Histogram(bins=11)
    h2.fit(X, sample_weight=w)

    assert_array_almost_equal(
        h1.pdf([[0.0], [1.0], [10.0], [-0.5], [10.5]]),
        h2.pdf([[0.0], [1.0], [10.0], [-0.5], [10.5]]))

    assert_raises(ValueError, h1.fit, X, sample_weight=w[1:])


def test_histogram_2d():
    X = np.arange(100).reshape(-1, 2)
    h = Histogram(bins=[5, 3])
    h.fit(X)
    assert h.ndim == 2
    assert h.histogram_.shape[0] == 5+2
    assert h.histogram_.shape[1] == 3+2


def test_histogram_variable_width():
    X = np.arange(11).reshape(-1, 1)
    h = Histogram(bins=11, variable_width=True)
    h.fit(X)
    assert_array_almost_equal(h.pdf([[1.0], [2.0], [8.0]]), [0.1, 0.1, 0.1])

    h = Histogram(bins=3, variable_width=True)
    h.fit(X)
    integral = h.histogram_ * (h.edges_[0][1:] - h.edges_[0][:-1])
    integral = integral[1:-1].sum()
    assert_almost_equal(integral, 1.)


def test_histogram_normalization():
    rng = check_random_state(1)
    X = rng.rand(100, 1)

    for h in [Histogram(bins=10),
              Histogram(bins=10, variable_width=True),
              Histogram(bins="blocks")]:
        h = Histogram(bins=10)
        h.fit(X)

        volume = ((h.edges_[0][2:-1] -
                   h.edges_[0][1:-2]) * h.histogram_[1:-1]).sum()
        assert_almost_equal(volume, 1.0)


def test_histogram_std():
    rng = check_random_state(1)
    X = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9]).reshape(-1, 1)
    h = Histogram(bins=5, range=[(0, 1)])
    h.fit(X)
    p, std = h.pdf(X, return_std=True)
    assert std[0] == 5 ** 0.5 / 6
    assert std[-1] == 1 ** 0.5 / 6
