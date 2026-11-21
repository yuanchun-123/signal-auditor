"""Render an :class:`AuditReport` as Markdown or a self-contained HTML page."""

from __future__ import annotations

import base64
import html as _html
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}


def render_markdown(rep) -> str:
    lines = [f"# Signal audit: {rep.name}", "", f"**Verdict: {rep.verdict}**  ",
             " · ".join(f"{k} {v}" for k, v in rep.counts().items()), "",
             "| | Check | Finding |", "|---|---|---|"]
    for r in rep.results:
        lines.append(f"| {ICON[r.status]} | {r.name} | {r.summary} |")
    return "\n".join(lines) + "\n"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _charts(rep) -> list[tuple[str, str]]:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#666", "axes.labelcolor": "#666",
                         "xtick.color": "#666", "ytick.color": "#666", "axes.edgecolor": "#999"})
    out = []

    fig, ax = plt.subplots(figsize=(6, 2.6))
    rep.ic.cumsum().plot(ax=ax, color="#3b6fd8", lw=1.6)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("cumulative rank IC")
    out.append(("Cumulative rank IC", _png(fig)))

    fig, ax = plt.subplots(figsize=(6, 2.6))
    qm = rep.quantile_returns.mean() * 1e4
    ax.bar([str(c) for c in qm.index], qm.values, color="#3b6fd8")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("signal quantile (1 = lowest)")
    ax.set_ylabel("mean fwd return (bps)")
    out.append(("Mean forward return by quantile", _png(fig)))

    fig, ax = plt.subplots(figsize=(6, 2.6))
    bt = rep.backtest
    bt["gross"].cumsum().plot(ax=ax, color="#3b6fd8", lw=1.6, label="gross")
    bt["net"].cumsum().plot(ax=ax, color="#d8813b", lw=1.6,
                            label=f"net of {rep.config.cost_bps:g} bps")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("cumulative L/S return")
    ax.legend(frameon=False)
    out.append(("Long-short, non-overlapping rebalances", _png(fig)))
    return out


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e4e4e2;
--pass:#1f8a4c;--warn:#b7791f;--fail:#c0392b;--passbg:#e8f5ee;--warnbg:#fdf3e1;--failbg:#fbe9e7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151517;--fg:#ececee;
--muted:#9a9aa0;--card:#1e1e21;--line:#2e2e33;--passbg:#16301f;--warnbg:#33290f;--failbg:#3a1a16}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 12px}
.muted{color:var(--muted)}.verdict{display:inline-block;margin:14px 0 4px;padding:6px 12px;
border-radius:8px;font-weight:600}
.v-NOT{background:var(--failbg);color:var(--fail)}.v-PROVISIONAL{background:var(--warnbg);color:var(--warn)}
.v-NO{background:var(--passbg);color:var(--pass)}
.check{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:8px 0}
.check summary{cursor:pointer;display:flex;gap:10px;align-items:baseline;list-style:none}
.badge{font-size:11px;font-weight:700;padding:2px 7px;border-radius:5px;flex:none}
.PASS{background:var(--passbg);color:var(--pass)}.WARN{background:var(--warnbg);color:var(--warn)}
.FAIL{background:var(--failbg);color:var(--fail)}
.check pre{font-size:12px;overflow-x:auto;margin:10px 0 0;color:var(--muted);white-space:pre-wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
figure img{width:100%;height:auto}figcaption{font-size:13px;color:var(--muted);margin-bottom:6px}
"""


def render_html(rep) -> str:
    esc = _html.escape
    c = rep.counts()
    vclass = rep.verdict.split()[0]
    checks = []
    for r in rep.results:
        det = json.dumps(r.details, indent=2, default=str)
        checks.append(
            f'<details class="check"><summary><span class="badge {r.status}">{r.status}</span>'
            f"<span><b>{esc(r.name)}</b> — {esc(r.summary)}</span></summary>"
            f"<pre>{esc(det)}</pre></details>")
    figs = "".join(f'<figure><figcaption>{esc(t)}</figcaption>'
                   f'<img alt="{esc(t)}" src="data:image/png;base64,{b}"></figure>'
                   for t, b in _charts(rep))
    cfg = rep.config
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Signal Audit Report</title><style>{CSS}</style></head><body><main>
<h1>Signal audit: {esc(rep.name)}</h1>
<div class="muted">horizon {cfg.horizon} · {cfg.periods_per_year:g} periods/yr ·
{cfg.n_trials} trial(s) declared · {cfg.cost_bps:g} bps one-way cost</div>
<div class="verdict v-{vclass}">Verdict: {esc(rep.verdict)}</div>
<div class="muted">{c['PASS']} pass · {c['WARN']} warn · {c['FAIL']} fail.
A clean report means no known failure mode was detected — not that the signal is alpha.</div>
<h2>Checks</h2>{''.join(checks)}
<h2>Diagnostics</h2><div class="grid">{figs}</div>
</main></body></html>"""
