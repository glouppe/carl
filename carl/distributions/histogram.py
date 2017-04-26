# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from astropy.stats import bayesian_blocks
from itertools import product
from scipy.interpolate import interp1d
from sklearn.utils import check_random_state
from sklearn.utils import check_array

from .base import DistributionMixin


class Histogram(DistributionMixin):
    """N-dimensional histogram."""

    def __init__(self, bins=10, range=None, interpolation=None,
                 variable_width=False):
        """Constructor.

        Parameters
        ----------
        * `bins` [int or string]:
            The number of bins or `"blocks"` for bayesian blocks.

        * `range` [list of bounds (low, high)]:
            The boundaries. If `None`, bounds are inferred from data.

        * `interpolation` [string, optional]
            Specifies the kind of interpolation between bins as a string
            (`"linear"`, `"nearest"`, `"zero"`, `"slinear"`, `"quadratic"`,
            `"cubic"`).

        * `variable_width` [boolean, optional]
            If True use equal probability variable length bins.
        """
        self.bins = bins
        self.range = range
        self.interpolation = interpolation
        self.variable_width = variable_width

    def pdf(self, X, return_std=False, **kwargs):
        X = check_array(X)

        if self.ndim_ == 1 and self.interpolation:
            return self.interpolation_(X[:, 0])

        all_indices = []

        for j in range(X.shape[1]):
            indices = np.searchsorted(self.edges_[j],
                                      X[:, j],
                                      side="right") - 1

            # For the last bin, the upper is inclusive
            indices[X[:, j] == self.edges_[j][-2]] -= 1
            all_indices.append(indices)

        p = self.histogram_[all_indices]

        if return_std:
            return p, self.errors_[all_indices]
        else:
            return p

    def nll(self, X, return_std=False, **kwargs):
        if not return_std:
            p = self.pdf(X, **kwargs)
        else:
            p, std = self.pdf(X, return_std=True, **kwargs)

        if not return_std:
            return -np.log(p)
        else:
            s = std / p
            s[~np.isfinite(s)] = 0

            return -np.log(p), s


    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)

        # Draw random bins with respect to their densities
        h = (self.histogram_ / self.histogram_.sum()).ravel()
        flat_indices = np.searchsorted(np.cumsum(h),
                                       rng.rand(n_samples),
                                       side="right")

        # Build bin corners
        indices = np.unravel_index(flat_indices, self.histogram_.shape)
        indices_end = [a + 1 for a in indices]
        shape = [len(d) for d in self.edges_] + [len(self.edges_)]
        corners = np.array(list(product(*self.edges_))).reshape(shape)

        # Draw uniformly within bins
        low = corners[indices]
        high = corners[indices_end]
        u = rng.rand(*low.shape)

        return low + u * (high - low)

    def fit(self, X, sample_weight=None, **kwargs):
        # Checks
        X = check_array(X)

        if sample_weight is not None and len(sample_weight) != len(X):
            raise ValueError
        if (self.bins == "blocks" or
                self.variable_width) and (X.shape[1] != 1):
            raise ValueError

        # Compute histogram and edges
        if self.bins == "blocks":
            bins = bayesian_blocks(X.ravel(), fitness="events", p0=0.0001)
            range_ = self.range[0] if self.range else None
            h, e = np.histogram(X.ravel(), bins=bins, range=range_,
                                weights=sample_weight, normed=False)
            counts = h
            volumes = e[1:] - e[:-1]
            e = [e]

        elif self.variable_width:
            ticks = [np.percentile(X.ravel(), 100. * k / self.bins) for k
                     in range(self.bins + 1)]
            ticks[-1] += 1e-5
            range_ = self.range[0] if self.range else None
            h, e = np.histogram(X.ravel(), bins=ticks, range=range_,
                                normed=False, weights=sample_weight)
            h, e = h.astype(float), e.astype(float)
            counts = h
            volumes = e[1:] - e[:-1]
            e = [e]

        else:
            bins = self.bins
            h, e = np.histogramdd(X, bins=bins, range=self.range,
                                  weights=sample_weight, normed=False)
            counts = h
            volumes = np.ones_like(h)
            for e_i in np.meshgrid(*[e_i[1:] - e_i[:-1] for e_i in e],
                                   indexing="ij"):
                volumes *= e_i

        # Histogram and bin uncertainties
        h = counts / counts.sum() / volumes
        errors = np.sqrt(counts) / counts.sum()  # Poisson errors

        # Add empty bins for out of bound samples
        for j in range(X.shape[1]):
            h = np.insert(h, 0, 0., axis=j)
            h = np.insert(h, h.shape[j], 0., axis=j)
            errors = np.insert(errors, 0, 0., axis=j)
            errors = np.insert(errors, errors.shape[j], 0., axis=j)
            e[j] = np.insert(e[j], 0, -np.inf)
            e[j] = np.insert(e[j], len(e[j]), np.inf)

        if X.shape[1] == 1 and self.interpolation:
            inputs = e[0][2:-1] - (e[0][2] - e[0][1]) / 2.
            inputs[0] = e[0][1]
            inputs[-1] = e[0][-2]
            outputs = h[1:-1]
            self.interpolation_ = interp1d(inputs, outputs,
                                           kind=self.interpolation,
                                           bounds_error=False,
                                           fill_value=0.)

        self.histogram_ = h
        self.edges_ = e
        self.counts_ = counts
        self.errors_ = errors
        self.ndim_ = X.shape[1]
        return self

    def cdf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def ppf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    @property
    def ndim(self):
        return self.ndim_
