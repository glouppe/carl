"""This module implements classifier-based density ratio estimation."""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.utils import check_random_state

from .base import DensityRatioMixin
from ..learning import as_classifier


class ClassifierRatio(BaseEstimator, DensityRatioMixin):
    """Classifier-based density ratio estimator.

    This class approximates a density ratio `r(x) = p0(x) / p1(x)` as
    `s(x) / 1 - s(x)`, where `s` is a classifier trained to distinguish
    samples `x ~ p0` from samples `x ~ p1`, and where `s(x)` is the
    classifier approximate of the probability `p0(x) / (p0(x) + p1(x))`.

    This class can be used in the likelihood-free setup, i.e. either

    - with known data `X` drawn from `p0` and `p1`, or
    - with generators `p0` and `p1` implementing sampling through `rvs`.
    """

    def __init__(self, base_estimator, random_state=None):
        """Constructor.

        Parameters
        ----------
        * `base_estimator` [`BaseEstimator`]:
            A scikit-learn classifier or regressor.

        * `random_state` [integer or RandomState object]:
            The random seed.
        """
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(self, X=None, y=None, sample_weight=None,
            numerator=None, denominator=None, n_samples=None, **kwargs):
        """Fit the density ratio estimator.

        The density ratio estimator `r(x) = p0(x) / p1(x)` can be fit either

        - from data, using `fit(X, y)` or
        - from distributions, using
          `fit(numerator=p0, denominator=p1, n_samples=N)`

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features), optional]:
            Training data.

        * `y` [array-like, shape=(n_samples,), optional]:
            Labels. Samples labeled with `y=0` correspond to data from the
            numerator distribution, while samples labeled with `y=1` correspond
            data from the denominator distribution.

        * `sample_weight` [array-like, shape=(n_samples,), optional]:
            The sample weights.

        * `numerator` [`DistributionMixin`, optional]:
            The numerator distribution `p0`, if `X` and `y` are not provided.
            This object is required to implement sampling through the `rvs`
            method.

        * `denominator` [`DistributionMixin`, optional]:
            The denominator distribution `p1`, if `X` and `y` are not provided.
            This object is required to implement sampling through the `rvs`
            method.

        * `n_samples` [integer, optional]
            The total number of samples to draw from the numerator and
            denominator distributions, if `X` and `y` are not provided.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        # Check for identity
        self.identity_ = (numerator is not None) and (numerator is denominator)

        if self.identity_:
            return self

        # Build training data
        rng = check_random_state(self.random_state)

        if (numerator is not None and denominator is not None and
                n_samples is not None):
            X = np.vstack(
                (numerator.rvs(n_samples // 2,
                               random_state=rng, **kwargs),
                 denominator.rvs(n_samples - (n_samples // 2),
                                 random_state=rng, **kwargs)))
            y = np.zeros(n_samples, dtype=np.int)
            y[n_samples // 2:] = 1
            sample_weight = None

        elif X is not None and y is not None:
            if sample_weight is None:
                n_num = (y == 0).sum()
                n_den = (y == 1).sum()

                if n_num != n_den:
                    sample_weight = np.ones(len(y), dtype=np.float)
                    sample_weight[y == 1] *= 1.0 * n_num / n_den
                else:
                    sample_weight = None

        else:
            raise ValueError

        # Fit base estimator
        clf = clone(self.base_estimator)

        if isinstance(clf, RegressorMixin):
            clf = as_classifier(clf)

        if sample_weight is None:
            self.classifier_ = clf.fit(X, y)
        else:
            self.classifier_ = clf.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X, return_std=False, log=False, **kwargs):
        """Predict the density ratio `r(x_i)` for all `x_i` in `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        * `log` [boolean, default=False]:
            If true, return the log-ratio `log r(x) = log(p0(x)) - log(p1(x))`.

        Returns
        -------
        * `r` [array, shape=(n_samples,)]:
            The predicted ratio `r(X)`.
        """
        if self.identity_:
            if not return_std:
                if log:
                    return np.zeros(len(X))
                else:
                    return np.ones(len(X))
            else:
                if log:
                    return np.zeros(len(X)), np.zeros(len(X))
                else:
                    return np.ones(len(X)), np.zeros(len(X))

        else:
            if not return_std:
                p = self.classifier_.predict_proba(X)

                if log:
                    return np.log(p[:, 0]) - np.log(p[:, 1])
                else:
                    return np.divide(p[:, 0], p[:, 1])

            else:
                p, std_p = self.classifier_.predict_proba(X, return_std=True)

                r = np.divide(p[:, 0], p[:, 1])
                std_r = (r ** 2 * ((std_p / p[:, 0]) ** 2 +
                                   (std_p / p[:, 1]) ** 2 -
                                   2 * (-std_p ** 2) / (p[:, 0] *
                                                        p[:, 1]))) ** 0.5
                # nb: cov(p, 1-p) = -var(p) = -std^2(p)

                std_r[~np.isfinite(std_r)] = 0

                if not log:
                    return r, std_r

                else:
                    std_r = std_r / r
                    std_r[~np.isfinite(std_r)] = 0

                    return np.log(p[:, 0]) - np.log(p[:, 1]), std_r
