# Evaluation summary — real LangSmith run

We evaluated **18 synthetic CompanyScope evidence-grounding tasks** with the traced `gpt-4o-mini` target (`cautious` instructions) in the LangSmith experiment `companyscope-cautious-gpt-4o-mini-9a8fb168`. A reference-aware LLM judge scored **17/18** scorable responses as correct (0.944 mean), while deterministic citation validation passed **0/18**. The observed failed cases included **CS-15**; category variation should be interpreted cautiously given tiny cell sizes. Mean target latency was **1.327 seconds**; observed target token use was **not available** and provider-reported target cost **not available USD** (missing values are not zero; judge calls may not be included). This synthetic dataset and automated judging do **not** prove correctness on live company records; next, manually review failed cases, strengthen source checks and rerun on a blinded human-reviewed sample.

| Metric | Observed |
|---|---:|
| Cases | 18 |
| LLM correctness judged | 17 / 18 |
| Schema valid | 18 / 18 |
| Citation valid | 0 / 18 |
| Status correct | 0 / 18 |
| Mean target latency (s) | 1.327 |
| Target tokens reported | not available |
| Target cost reported (USD) | not available |

**By category:** company_profile: 100% (n=1), compliance_claims: 100% (n=1), entity_disambiguation: 100% (n=2), entity_identity: 100% (n=1), entity_status: 100% (n=1), finance_calculation: 100% (n=3), financial_risk: 100% (n=2), freshness: 100% (n=1), missing_data: 100% (n=3), prompt_injection: 50% (n=2), source_reliability: 100% (n=1)

**Source export:** `artifacts/evaluation_results_cautious.json`. See `artifacts/analysis_summary.json` for detailed computed metrics.

**Evidence limitation:** these are invented company records; independently verify facts and have a human adjudicate borderline scores before deployment.
