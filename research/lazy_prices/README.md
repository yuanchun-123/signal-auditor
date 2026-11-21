# Lazy Prices on public data, audited

A first-pass replication of Cohen, Malloy & Nguyen (2020), *"Lazy Prices"*
(Journal of Finance), built only from free data and run through `signal-auditor`.

**Hypothesis.** Firms that substantially change the text of their annual report
("changers") subsequently underperform firms whose report barely changes, because
investors are slow to read the changes.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_universe_filings.py` | Universe = 500 largest SEC registrants by public float **as of 2010–2011** (XBRL `dei:EntityPublicFloat`), not today's index; lists their 10-K filings 2010–2026 |
| 2 | `step2_similarity.py` | Downloads each 10-K's primary document (≈7,200 filings), reduces it to a bag of words, and computes year-over-year cosine and Jaccard similarity with the previous 10-K |
| 3 | `step3_panel_audit.py` | Monthly panel 2013-01 … 2026-09: each month-end carries the similarity from the latest 10-K tradeable by then; forward 1-month returns from Yahoo Finance; audit with Fama-French 5 + momentum spanning and size / momentum / volatility neutralisation |

```bash
export SEC_USER_AGENT="Your Name your.email@domain"   # required by SEC
python step1_universe_filings.py      # ~3 min
python step2_similarity.py            # ~25 min, resumable
python step3_panel_audit.py           # ~15 s once prices are cached
```

## Results (283 firms × 165 months)

| Run | Mean rank IC | NW t | Verdict | Main reasons |
|---|---|---|---|---|
| Cosine similarity, correct dating | 0.004 | 0.74 | NOT SUPPORTED | no predictive power; break-even cost 3 bps |
| Jaccard similarity, correct dating | 0.012 | 2.08 | NOT SUPPORTED | t below the bar of 3; spread comes entirely from the top quintile; alpha t = 1.46 after FF5 + momentum |
| Cosine similarity, **naive dating** | 0.002 | 0.33 | NOT SUPPORTED | **timestamp check: 16.7% of rows use a 10-K before it was filed** |

Jaccard quintile returns (monthly, low → high similarity): 1.15%, 1.21%, 1.13%, 1.08%, 1.42%.
The sign matches the paper (non-changers do better), but the effect is not
monotone and does not survive a multiple-testing-aware bar. Among the 500
largest firms over 2013–2026 there is, at best, weak evidence for the effect
in annual reports.

## What the audit caught along the way

1. **Fiscal-year-end dating is look-ahead.** Dating the signal at the fiscal
   period end (`reportDate`) instead of SEC acceptance uses information
   60–90 days early in 16.7% of firm-months. The timestamp check flags it.
2. **After-close filings.** 56% of 10-Ks are accepted at or after the 16:00 ET
   close. A filing accepted after the close on the last day of a month cannot be
   traded at that month-end close. A first version of the pipeline treated them
   as tradeable the same day; the timestamp check flagged 234 rows.
3. **Time zones.** EDGAR's `acceptanceDateTime` is UTC. Read as Eastern time,
   80% of filings appeared to arrive after the close and evening filings
   shifted to the wrong day.
4. **Dirty XBRL.** Self-reported public float contains typos (1e18 USD, or
   values 1000× too large). The universe takes the smaller of the 2010 and 2011
   values and drops floats above 20× total assets.
5. **Survivorship.** 156 of the 500 firms no longer have a ticker (mostly
   acquired or delisted), so free price data cannot follow them. Only 283 firms
   reach the panel. Passing `intended_assets=500` makes the auditor warn about this.

## Limitations and next steps

- **Prices:** Yahoo covers survivors only. With CRSP (WRDS), including delisting
  returns, the dropped firms can be restored. This is the most important fix.
- **Text:** the paper also uses 10-Qs and finds stronger effects in smaller
  firms; this run uses 10-Ks for large caps only. Raw-count cosine similarity is
  dominated by stop-words (median 0.999); removing stop-words or using TF-IDF is
  a natural refinement, but it must be counted as an additional trial.
- **Sections:** changes in specific sections (risk factors, legal proceedings,
  accounting policies) are where the paper finds the strongest effects. An
  LLM classifier of *what* changed is the planned extension.
