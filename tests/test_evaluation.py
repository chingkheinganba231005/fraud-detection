import numpy as np
import pytest

from fraud import evaluation


def test_at_budget_counts_hits_and_value():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 1])
    s = np.array([9, 8, 7, 1, 2, 3, 4, 0], dtype=float)
    v = np.arange(8, dtype=float)
    r = evaluation.at_budget(y, s, 3, v)
    assert r["hits"] == 2 and r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["value_found"] == pytest.approx(0 + 2)
    assert r["lift"] == pytest.approx((2 / 3) / (3 / 8))


def test_ties_are_not_broken_by_row_order():
    s = np.zeros(1000)
    first = evaluation.top_k(s, 10)
    assert not np.array_equal(np.sort(first), np.arange(10))


def test_bootstrap_intervals_contain_the_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(60), np.zeros(20000)].astype(int)
    s = np.r_[rng.normal(2, 1, 60), rng.normal(0, 1, 20000)]
    point = evaluation.summary(y, s, [100])
    ci = evaluation.bootstrap(y, s, [100], 500, 0)
    assert ci["roc_auc_ci"][0] <= point["roc_auc"] <= ci["roc_auc_ci"][1]
    assert ci["average_precision_ci"][0] <= point["average_precision"] <= ci["average_precision_ci"][1]
    rec = point["budgets"]["100"]["recall"]
    assert ci["recall_ci"]["100"][0] <= rec <= ci["recall_ci"]["100"][1]


def test_wilson_interval():
    lo, hi = evaluation.wilson(5, 100)
    assert 0 < lo < 0.05 < hi < 0.12
