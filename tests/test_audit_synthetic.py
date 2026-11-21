"""The auditor must recover the known answer on planted signals."""

import pandas as pd
import pytest

from signal_auditor import FAIL, PASS, audit
from signal_auditor.panel import forward_returns, to_long, validate_panel
from signal_auditor.synthetic import make_synthetic


@pytest.fixture(scope="module")
def panels():
    return make_synthetic(n_dates=180, n_assets=200, ic=0.04, n_random=100, seed=11)


def run(panel, **kw):
    kw.setdefault("periods_per_year", 12)
    kw.setdefault("n_placebo", 200)
    return audit(panel, **kw)


def test_true_signal_passes(panels):
    rep = run(panels["true"])
    assert rep.verdict == "NO RED FLAGS DETECTED", rep.to_markdown()


def test_leaky_signal_fails_delay_test(panels):
    rep = run(panels["leaky"])
    assert rep.status_of("Leakage") == FAIL
    assert rep.verdict == "NOT SUPPORTED"


def test_late_timestamps_fail(panels):
    assert run(panels["stale_timestamp"]).status_of("Point-in-time") == FAIL


def test_mined_signal_fails_once_trials_declared(panels):
    honest = run(panels["mined"], n_trials=100)
    assert honest.status_of("Deflated Sharpe") == FAIL
    assert honest.status_of("IC significance") != PASS
    assert honest.verdict == "NOT SUPPORTED"


def test_reports_render(panels, tmp_path):
    rep = run(panels["true"], n_placebo=50)
    html = rep.to_html(str(tmp_path / "r.html"))
    assert "Verdict" in html and (tmp_path / "r.html").exists()
    assert rep.to_markdown().startswith("# Signal audit")


def test_validate_panel_rejects_duplicates():
    df = pd.DataFrame({"date": ["2020-01-31"] * 2, "asset": ["A", "A"], "signal": [1, 2], "fwd_ret": [0, 0]})
    with pytest.raises(ValueError):
        validate_panel(df)


def test_unknown_option_rejected(panels):
    with pytest.raises(TypeError):
        audit(panels["true"], horizon_days=5)


def test_forward_returns_and_to_long():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 25.0]},
                          index=pd.date_range("2020-01-01", periods=3))
    fwd = forward_returns(prices, 1)
    assert fwd.loc["2020-01-01", "A"] == pytest.approx(0.10)
    assert fwd.loc["2020-01-02", "B"] == pytest.approx(-0.5)
    long = to_long(prices, fwd)
    assert len(long) == 4  # last date has no forward return


def test_universe_coverage_warns(panels):
    rep = run(panels["true"], intended_assets=1000, n_placebo=50)
    assert rep.status_of("Universe") == "WARN"


def test_negative_spread_fails_costs(panels):
    flipped = panels["true"].assign(signal=lambda d: -d["signal"])
    rep = run(flipped, n_placebo=50)
    assert rep.status_of("Transaction costs") == FAIL
    assert "negative before costs" in next(r.summary for r in rep.results
                                           if r.name == "Transaction costs")
