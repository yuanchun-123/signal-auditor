import numpy as np
import pandas as pd
import pytest

from signal_auditor import FAIL, PASS, audit
from signal_auditor import metrics as M
from signal_auditor.factors import _parse_french_csv, align_to_rebalances
from signal_auditor.synthetic import make_factor_case


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = 0.5 + 2.0 * x + 0.1 * rng.standard_normal(2000)
    fit = M.ols_newey_west(y, np.column_stack([np.ones_like(x), x]), lags=3)
    assert fit["coef"] == pytest.approx([0.5, 2.0], abs=0.01)
    assert fit["r2"] > 0.99


def test_ols_lag0_is_hc0():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(300), rng.standard_normal(300)])
    y = X @ [1.0, -1.0] + rng.standard_normal(300) * (1 + np.abs(X[:, 1]))
    fit = M.ols_newey_west(y, X, lags=0)
    xtx_inv = np.linalg.inv(X.T @ X)
    e = y - X @ fit["coef"]
    hc0 = xtx_inv @ (X.T * e**2) @ X @ xtx_inv
    assert fit["se"] == pytest.approx(np.sqrt(np.diag(hc0)))


def test_ols_constant_only_matches_newey_west_mean():
    x = np.random.default_rng(2).standard_normal(400)
    fit = M.ols_newey_west(x, np.ones((400, 1)), lags=4)
    assert fit["se"][0] == pytest.approx(M.newey_west_se(x, 4))


def test_parse_french_monthly():
    text = ("This file was created ...\n\n,Mkt-RF,SMB,RF\n196307,   -0.39,   -0.48,    0.27\n"
            "196308,    5.07,   -0.80,    0.25\n\n Annual Factors: January-December \n,Mkt-RF,SMB,RF\n"
            "1964,  10.0, 1.0, 3.0\n")
    df = _parse_french_csv(text, "M")
    assert list(df.columns) == ["Mkt-RF", "SMB", "RF"]
    assert df.index[0] == pd.Timestamp("1963-07-31")
    assert len(df) == 2 and df.iloc[1, 0] == pytest.approx(0.0507)


def test_align_uses_following_periods():
    cal = pd.date_range("2020-01-31", periods=6, freq="ME")
    f = pd.DataFrame({"F": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=cal)
    out = align_to_rebalances(f, cal, cal[[0, 2]], horizon=2)
    assert out.loc[cal[0], "F"] == 2.0 + 3.0     # earned over (Jan, Mar]
    assert out.loc[cal[2], "F"] == 4.0 + 5.0
    tail = align_to_rebalances(f, cal, cal[[5]], horizon=1)
    assert tail.isna().all().all()                # no future factor data -> NaN


@pytest.fixture(scope="module")
def factor_case():
    return make_factor_case(n_dates=180, n_assets=200, seed=5)


def test_disguised_factor_is_caught(factor_case):
    panels, factors = factor_case
    rep = audit(panels["disguised"], periods_per_year=12, factors=factors,
                neutralize=["char"], n_placebo=50)
    assert rep.status_of("Factor spanning") == FAIL
    assert rep.status_of("Characteristic-neutral") == FAIL


def test_genuine_alpha_survives(factor_case):
    panels, factors = factor_case
    rep = audit(panels["genuine"], periods_per_year=12, factors=factors,
                neutralize=["char"], n_placebo=50)
    assert rep.status_of("Factor spanning") == PASS
    assert rep.status_of("Characteristic-neutral") == PASS
    assert rep.to_dict()["config"]["factors"] == ["F"]
