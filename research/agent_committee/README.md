# An LLM investment committee, audited

Persona agents in the style of well-known investors (quality compounder, deep
value, GARP, magic formula, trend follower, forensic accountant) each score a
cross-section of US large caps every quarter, debate the stocks they disagree
on, and the resulting signals are run through `signal-auditor`.

The point is not another agent that "beats the market". It is a test bench for
four questions the literature rarely answers (see [LITERATURE.md](LITERATURE.md)):

| Question | How it is measured |
|---|---|
| Is a persona just a factor tilt? | Each persona has a **rule-based twin**: its stated rules as a formula. The LLM score is audited against FF5 + momentum and neutralised on size, value, momentum **and its own twin**. |
| Does debate add anything beyond a vote? | Round-0 mean (a vote) vs final-round mean (after debate) on the same disputed firms; conformity diagnostics. |
| Is it memorisation? | Firms are anonymised: no names, tickers, dates or exact size; IDs are re-randomised every quarter. `cards.identity_probe_prompt` measures how often a model can still name the company. |
| Does it survive costs and multiple testing? | Every persona, twin and aggregate counts as a trial; 10 bps one-way costs; deflated Sharpe. |

## Pipeline

| File | Role |
|---|---|
| `data_build.py` | Point-in-time features per quarter-end 2014-12 … 2026-06: annual XBRL facts as first filed (SEC companyfacts, `filed` ≤ date), split-unadjusted market value, momentum and volatility; next-quarter returns |
| `cards.py` | Anonymised fact sheets (24 ratios), random per-quarter IDs, chunks of ~40 firms |
| `personas.py` | Six personas with philosophies and rule-based twins; `Persona.from_skill_md` loads a GitHub `SKILL.md` persona |
| `committee.py` | Independent round, debate on the most disputed firms, conformity diagnostics, committee signals |
| `llm.py` | Claude backend: structured JSON output, sync with refusal fallbacks or Message Batches (50% cheaper), JSONL cache so runs resume |
| `evaluate.py` | Audit of every signal; debate vs vote on debated firms |
| `run.py` | Command line |

```bash
python data_build.py                                   # needs SEC_USER_AGENT or ~/.sec_user_agent
python run.py --backend rule                           # no API key: twins + simulated debate
python run.py --backend claude --dates 4 --estimate    # cost estimate only
python run.py --backend claude --dates 4               # pilot
python run.py --backend claude --mode batch            # full sample, batch pricing
```

The Claude backend reads `ANTHROPIC_API_KEY`. The default model is
`claude-opus-5-5`; `--model claude-sonnet-5-5` or `claude-haiku-5-5` are
cheaper. Run `--estimate` first.

## Known limitations

- Same universe as the Lazy Prices study: 500 largest firms by 2010–11 public
  float, restricted to those with prices today (survivorship; CRSP would fix it).
- Annual fundamentals only (10-K), no quarterly updates, no news or text, so the
  agents see less than AlphaAgents' sentiment agent would.
- Anonymisation reduces but does not remove memorisation risk: distinctive
  ratio patterns can still identify mega-caps. Run the identity probe.
