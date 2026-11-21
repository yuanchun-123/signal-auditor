"""Audit four planted signals whose true nature is known.

    python examples/synthetic_demo.py

Writes one HTML report per case to ``examples/reports/``.
"""

from pathlib import Path

from signal_auditor import audit
from signal_auditor.synthetic import make_synthetic

OUT = Path(__file__).parent / "reports"
OUT.mkdir(exist_ok=True)

CASES = [
    ("true", 1, "Genuine persistent signal (IC ~ 0.03)"),
    ("leaky", 1, "Signal contaminated with its own forward return"),
    ("mined", 1, "Best of 200 random signals -- trials NOT declared"),
    ("mined", 200, "Best of 200 random signals -- 200 trials declared"),
    ("stale_timestamp", 1, "Genuine signal, 15% of values published after use"),
]

panels = make_synthetic()
for key, trials, label in CASES:
    rep = audit(panels[key], name=label, periods_per_year=12, n_trials=trials)
    path = OUT / f"synthetic_{key}_trials{trials}.html"
    rep.to_html(str(path))
    print(f"{rep.verdict:22s}  {label}  ->  {path.name}")
