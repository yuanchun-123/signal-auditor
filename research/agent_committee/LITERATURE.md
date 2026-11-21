# Literature and open-source map (as of November 2026)

Notes for the agent-committee project. "Claim" = what the authors report; the
"Caveat" column is ours. Verify details against the papers before citing.

## 1. Multi-agent LLM investment systems

| Work | Design | Claim | Caveat |
|---|---|---|---|
| **AlphaAgents** (Zhao, Lyu, Jones, Garber, Pasquali, Mehta; BlackRock), [arXiv 2508.11152](https://arxiv.org/abs/2508.11152) | Fundamental (10-K/10-Q), sentiment (news) and valuation (prices) agents; debate only on disagreement; moderator + selector agents; risk-tolerance prompts | Multi-agent portfolio beats single agents and the benchmark | The authors' own limitations slide: hallucination, no transaction costs, backtest window limited by the knowledge cut-off, agents converge in 68% of debates |
| **TradingAgents** (Xiao et al.), [arXiv 2412.20138](https://arxiv.org/abs/2412.20138), [GitHub](https://github.com/TauricResearch/TradingAgents) | Analysts, bull/bear researchers who debate, trader, risk team, portfolio manager (LangGraph) | Better cumulative return and Sharpe than baselines on a few stocks | Single stocks, short in-knowledge windows |
| **FinCon** (Yu et al., NeurIPS 2024), [arXiv 2407.06567](https://arxiv.org/abs/2407.06567), [GitHub](https://github.com/The-FinAI/FinCon) | Manager-analyst hierarchy; risk control critiques performance and updates "investment beliefs" (verbal reinforcement) | Beats earlier agent systems on single-stock trading and portfolio tasks | Same evaluation-window issue |
| **FinPanel** — "Who Should Be on the Panel?", [arXiv 2611.18150](https://arxiv.org/abs/2611.18150) | A recruiter picks named trader personas per stock from a direction-blind statistical sketch; manager weighs their track records | On 8 stocks in the out-of-knowledge year 2025, the selected panel beats a fixed panel and prior agents | 8 stocks; the closest prior work to persona committees |
| **Multi-Agent Debate for Explainable Trading**, [arXiv 2609.29701](https://arxiv.org/abs/2609.29701) | Agents propose, critique, revise allocations; reasoning quality graded | Over 210 runs, reasoning quality has **no** relation to Sharpe (r = 0.07) | Good reasons ≠ good returns: why we audit returns, not rationales |
| **Survey**: Saha, Lyu, Saxena, Zhao, Mehta, *LLM Agents for Investment Management*, ICAIF 2025 ([SSRN 5447274](https://papers.ssrn.com/abstract=5447274)) | 10-stage agent workflow; application map (retrieval, research, trading, portfolio, risk, advisors, market simulation) | — | The taxonomy in the lecture slides |

## 2. Investor personas ("guru" agents and skills)

| Work | What it is | Caveat |
|---|---|---|
| **GuruAgents**, [arXiv 2510.01664](https://arxiv.org/abs/2510.01664), [GitHub](https://github.com/yejining99/GuruAgents) | Five prompt-guided "guru" agents on NASDAQ-100; the Buffett agent reports a 42.2% CAGR, Q4 2023–Q2 2025 | Window is inside the models' knowledge, a strong large-cap tech rally, no factor adjustment: the case for our controls |
| **virattt/ai-hedge-fund**, [GitHub](https://github.com/virattt/ai-hedge-fund) (~50k stars) | Buffett, Munger, Burry, Lynch, Wood … agents give a conviction in [-1, 1]; a portfolio manager aggregates | Educational proof of concept; no out-of-sample evaluation |
| **Persona-based ensembles**, [arXiv 2411.19515](https://arxiv.org/abs/2411.19515) | Personas predict stock/bond moves; mode ensemble | Works in rising-CPI regimes only |
| **Do LLM Personas Dream of Bull Markets?**, [arXiv 2411.05801](https://arxiv.org/abs/2411.05801); **Investor risk profiles of LLMs**, [arXiv 2603.09303](https://arxiv.org/abs/2603.09303) | Persona behaviour vs human investors | Behavioural, not return evaluation |
| **Claude Code skills on GitHub**: [agi-now/buffett-skills](https://awesome.ecosyste.ms/projects/github.com%2Fagi-now%2Fbuffett-skills), [VictorVVedtion/trading-skills](https://github.com/VictorVVedtion/trading-skills) (68 "trading legends"), [muxuuu/serenity-skill](https://raw.githubusercontent.com/muxuuu/serenity-skill/main/SKILL.md) (supply-chain bottleneck research, after the X investor "Serenity") | `SKILL.md` instruction files distilled from public letters, talks and posts | Prompt personas; fidelity unmeasured. Load with `Persona.from_skill_md` after reading the file |

## 3. Why naive backtests of LLM agents mislead

**Look-ahead / memorisation**
- Gao, Jiang & Yan, *Detecting Lookahead Bias in LLM Forecasts*, [arXiv 2512.23847](https://arxiv.org/abs/2512.23847): a "lookahead propensity" test; predictive power is amplified where the model has memorised the outcome and disappears after the training cut-off.
- *Look-Ahead-Bench*, [arXiv 2601.13770](https://arxiv.org/abs/2601.13770): most published LLM agents fail to beat baselines after their knowledge cut-off.
- *Chronologically consistent* LLMs, [arXiv 2502.21206](https://arxiv.org/abs/2502.21206), and DatedGPT, [arXiv 2603.11838](https://arxiv.org/abs/2603.11838): time-restricted pre-training; the bias in news-based prediction is modest in some settings.
- Glasserman & Lin (2023): anonymising company identifiers helps; a "distraction effect" can outweigh look-ahead.

**Debate and conformity**
- *Talk Isn't Always Cheap*, [arXiv 2509.05396](https://arxiv.org/abs/2509.05396): debate can do worse than majority voting and deteriorate over rounds.
- *An Empirical Study of Group Conformity in Multi-Agent Systems*, [arXiv 2506.01332](https://arxiv.org/abs/2506.01332); *Free-MAD*, [arXiv 2509.11035](https://arxiv.org/abs/2509.11035) (anti-conformity, consensus-free).
- Several 2026 preprints argue that most of debate's gain is majority voting (to verify before citing).
- *Attention Tax, Handoff Tax*, [arXiv 2610.06069](https://arxiv.org/abs/2610.06069): when multi-agent systems beat a single agent with the same information.

**Robustness**
- *Contagion on the Trading Floor*, [arXiv 2609.19789](https://arxiv.org/abs/2609.19789): poisoned social-media inputs propagate through multi-agent trading stacks.

**Live / contamination-free benchmarks**
- StockBench [arXiv 2510.02209](https://arxiv.org/abs/2510.02209), Agent Market Arena [arXiv 2510.11695](https://arxiv.org/abs/2510.11695), LiveTradeBench [arXiv 2511.03628](https://arxiv.org/abs/2511.03628): the agent framework matters more than the backbone model; leaderboard rank does not predict trading results.

## 4. The gap this project targets

Published persona and committee systems rarely ask:

1. **Is a persona just a factor tilt?** A "Buffett" agent that likes high ROE, high margins and low debt is a quality factor. Without factor spanning and neutralisation on its own rules, a good backtest says nothing about the LLM.
2. **Does debate add anything beyond a vote?** Compare the post-debate mean with the pre-debate mean *on the same firms*, and measure conformity.
3. **Is the result memorisation?** Anonymise firms and dates, probe whether the model can still name the company, and keep the claims inside what survives.
4. **Does it survive costs and multiple testing?** Every persona and aggregate is a trial.
