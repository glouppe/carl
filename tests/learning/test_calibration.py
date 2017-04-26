# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises
from sklearn.utils.testing import assert_greater

from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import brier_score_loss

from carl.distributions import Normal
from carl.learning import CalibratedClassifierCV
from carl.learning.calibration import HistogramCalibrator


def check_calibration(method):
    # Adpated from sklearn/tests/test_calibration.py
    # Authors: Alexandre Gramfort
    # License: BSD 3 clause

    n_samples = 100
    X, y = make_classification(n_samples=2 * n_samples, n_features=6,
                               random_state=42)

    X -= X.min()  # MultinomialNB only allows positive X

    # split train and test
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Naive-Bayes
    clf = MultinomialNB().fit(X_train, y_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]

    pc_clf = CalibratedClassifierCV(clf, cv=y.size + 1)
    assert_raises(ValueError, pc_clf.fit, X, y)

    pc_clf = CalibratedClassifierCV(clf, method=method, cv=2)
    # Note that this fit overwrites the fit on the entire training set
    pc_clf.fit(X_train, y_train)
    prob_pos_pc_clf = pc_clf.predict_proba(X_test)[:, 1]

    # Check that brier score has improved after calibration
    assert_greater(brier_score_loss(y_test, prob_pos_clf),
                   brier_score_loss(y_test, prob_pos_pc_clf))

    # Check invariance against relabeling [0, 1] -> [1, 2]
    pc_clf.fit(X_train, y_train + 1)
    prob_pos_pc_clf_relabeled = pc_clf.predict_proba(X_test)[:, 1]
    assert_array_almost_equal(prob_pos_pc_clf,
                              prob_pos_pc_clf_relabeled)

    # Check invariance against relabeling [0, 1] -> [-1, 1]
    pc_clf.fit(X_train, 2 * y_train - 1)
    prob_pos_pc_clf_relabeled = pc_clf.predict_proba(X_test)[:, 1]
    assert_array_almost_equal(prob_pos_pc_clf,
                              prob_pos_pc_clf_relabeled)

    # Check invariance against relabeling [0, 1] -> [1, 0]
    pc_clf.fit(X_train, (y_train + 1) % 2)
    prob_pos_pc_clf_relabeled = pc_clf.predict_proba(X_test)[:, 1]
    if method == "sigmoid":
        assert_array_almost_equal(prob_pos_pc_clf,
                                  1 - prob_pos_pc_clf_relabeled)
    else:
        # Isotonic calibration is not invariant against relabeling
        # but should improve in both cases
        assert_greater(brier_score_loss(y_test, prob_pos_clf),
                       brier_score_loss((y_test + 1) % 2,
                                        prob_pos_pc_clf_relabeled))


def test_calibration():
    for method in ["isotonic", "sigmoid", "histogram", "kde",
                   "interpolated-isotonic"]:
        yield check_calibration, method


def test_calibration_histogram_std():
    p0 = Normal(mu=0.1, sigma=0.3)
    probas0 = p0.rvs(10000)
    probas0 = probas0[(probas0[:, 0] >= 0.0) & (probas0[:, 0] <= 1.0)]
    p1 = Normal(mu=0.9, sigma=0.3)
    probas1 = p1.rvs(10000)
    probas1 = probas1[(probas1[:, 0] >= 0.0) & (probas1[:, 0] <= 1.0)]

    X = np.vstack([probas0, probas1])
    y = np.zeros(len(X))
    y[len(probas0):] = 1

    h = HistogramCalibrator(bins=10)
    h.fit(X.ravel() / 2 + 0.25, y)

    xs = np.linspace(0.25, 0.75, 101).reshape(-1, 1)
    p, std = h.predict(xs.ravel(), return_std=True)

    assert std[50] > std[0]  # uncertainty should be higher near the boundary
    assert np.abs(std[0] - std[-1]) < 10e-5  # uncertainties should be similar


def test_calibration_clf_std():
    p0 = Normal(mu=0.1, sigma=0.3)
    probas0 = p0.rvs(10000)
    probas0 = probas0[(probas0[:, 0] >= 0.0) & (probas0[:, 0] <= 1.0)]
    p1 = Normal(mu=0.9, sigma=0.3)
    probas1 = p1.rvs(10000)
    probas1 = probas1[(probas1[:, 0] >= 0.0) & (probas1[:, 0] <= 1.0)]

    X = np.vstack([probas0, probas1])
    y = np.zeros(len(X))
    y[len(probas0):] = 1

    from sklearn.ensemble import ExtraTreesClassifier
    clf = CalibratedClassifierCV(ExtraTreesClassifier(max_leaf_nodes=5,
                                                      n_estimators=100),
                                 method="histogram", cv=3)
    clf.fit(X, y)

    xs = np.linspace(0, 1, 101).reshape(-1, 1)
    p, std = clf.predict_proba(xs, return_std=True)

    assert std[50] > std[0]  # uncertainty should be higher near the boundary
    assert np.abs(std[0] - std[-1]) < 10e-5  # uncertainties should be similar
