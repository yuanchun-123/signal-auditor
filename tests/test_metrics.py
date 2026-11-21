import math

import numpy as np
import pandas as pd
import pytest

from signal_auditor import metrics as M


def _panel(signal: np.ndarray, ret: np.ndarray) -> pd.DataFrame:
    t, n = signal.shape
    return pd.DataFrame({
        "date": np.repeat(pd.date_range("2020-01-31", periods=t, freq="ME"), n),
        "asset": np.tile(np.arange(n), t),
        "signal": signal.ravel(),
        "fwd_ret": ret.ravel(),
    })


def test_ic_perfect_and_inverse():
    x = np.tile(np.arange(10.0), (3, 1))
    assert M.ic_series(_panel(x, x)).eq(1.0).all()
    assert np.allclose(M.ic_series(_panel(x, -x)), -1.0)


def test_rank_ic_matches_pandas_spearman():
    rng = np.random.default_rng(0)
    s, r = rng.standard_normal((4, 50)), rng.standard_normal((4, 50))
    ic = M.ic_series(_panel(s, r), method="spearman")
    for i in range(4):
        assert ic.iloc[i] == pytest.approx(pd.Series(s[i]).corr(pd.Series(r[i]), method="spearman"))


def test_pearson_ic_matches_numpy():
    rng = np.random.default_rng(1)
    s, r = rng.standard_normal((3, 40)), rng.standard_normal((3, 40))
    ic = M.ic_series(_panel(s, r), method="pearson")
    for i in range(3):
        assert ic.iloc[i] == pytest.approx(np.corrcoef(s[i], r[i])[0, 1])


def test_newey_west_zero_lags_is_iid_se():
    x = np.random.default_rng(2).standard_normal(500)
    assert M.newey_west_se(x, 0) == pytest.approx(x.std(ddof=0) / math.sqrt(x.size))


def test_newey_west_inflates_se_for_overlapping_sums():
    # 12-period overlapping sums of iid noise are strongly autocorrelated
    e = np.random.default_rng(3).standard_normal(5000)
    x = np.convolve(e, np.ones(12), mode="valid")
    assert M.newey_west_se(x, 11) > 2.5 * M.newey_west_se(x, 0)


def test_newey_west_matches_hand_computation():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    e = x - x.mean()
    n = x.size
    g0 = e @ e / n
    g1 = e[1:] @ e[:-1] / n
    expected = math.sqrt((g0 + 2 * 0.5 * g1) / n)  # Bartlett weight 1 - 1/2
    assert M.newey_west_se(x, 1) == pytest.approx(expected)


def test_quantile_labels_balanced():
    x = np.tile(np.arange(100.0), (2, 1))
    q = M.quantile_labels(_panel(x, x), 5)
    assert q.value_counts().eq(40).all()


def test_long_short_weights_are_dollar_neutral():
    rng = np.random.default_rng(4)
    w = M.long_short_weights(_panel(rng.standard_normal((6, 50)), rng.standard_normal((6, 50))))
    assert np.allclose(w.sum(axis=1), 0.0)
    assert np.allclose(w.abs().sum(axis=1), 2.0)


def test_turnover_zero_for_constant_weights():
    w = pd.DataFrame([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    assert M.turnover(w).iloc[1:].eq(0).all()


def test_expected_max_sharpe_grows_with_trials():
    vals = [M.expected_max_sharpe(n, 0.1) for n in (1, 10, 100, 1000)]
    assert vals[0] == 0 and all(a < b for a, b in zip(vals, vals[1:]))


def test_deflated_sharpe_penalises_trials():
    r = np.random.default_rng(5).normal(0.01, 0.05, 240)
    one, *_ = M.deflated_sharpe_ratio(r, 1)
    many, *_ = M.deflated_sharpe_ratio(r, 500)
    assert many < one


def test_bonferroni_threshold():
    assert M.bonferroni_t_threshold(1) == pytest.approx(1.96, abs=1e-2)
    assert M.bonferroni_t_threshold(200) > 3.5
