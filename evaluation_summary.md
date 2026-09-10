# Evaluation Summary

We evaluated a `gpt-4o-mini` target function (temperature 0) that classifies raw, EU-style
bank-statement transaction lines into one of 15 categories and normalizes a merchant name,
against a 20-example [LangSmith dataset](https://eu.smith.langchain.com/datasets/b0a0b9b0-62f4-403b-ac35-3a4010741091)
of synthetic but realistic transactions (`bank-transaction-categorization-v1`), scored with two
evaluators: a deterministic `category_exact_match` check and an LLM-as-judge `correctness`
evaluator (also `gpt-4o-mini`) that accepts reasonable merchant-name variations. The model got
the **category exactly right 95% of the time (19/20)** but LLM-judged overall **correctness was
only 75% (15/20)** — the gap is almost entirely a merchant-name normalization problem: on 4 of
the 5 failures the category was correct but the model left raw statement noise in the merchant
field (e.g. `"AMAZON.DE MARKETPLACE"` instead of `"Amazon"`, `"MIETE OKTOBER WOHNUNG HAUPTSTR"`
instead of a clean building/landlord name) rather than actually normalizing it; the one true
category miss was `NETFLIX.COM` classified as `Subscriptions` when the reference expected
`Entertainment`, a genuinely ambiguous taxonomy boundary. Limitations: only 20 examples, all
synthetic/German-EU-style text (no real bank data, no other locales), one model/temperature
tested, and the correctness judge's own accept-reasonable-variation instruction is itself
subjective. Recommendation: tighten the target function's system prompt with explicit
merchant-normalization examples (strip suffixes like "MARKETPLACE", "GMBH", city names) and
either merge `Entertainment` into `Subscriptions` or add a disambiguation rule for streaming
services, then re-run to confirm the correctness rate closes toward the 95% exact-match rate.

## Key metrics

| Metric | Value |
|---|---|
| Examples evaluated | 20 |
| Category exact-match rate | 95% (19/20) |
| LLM-judged correctness rate | 75% (15/20) |
| Correctness by difficulty | easy 73% (8/11), medium 71% (5/7), hard 100% (2/2) |

## Links

- Dataset: https://eu.smith.langchain.com/datasets/b0a0b9b0-62f4-403b-ac35-3a4010741091
- Experiment (`txn-categorization-gpt-4o-mini-262d9e4d`): https://eu.smith.langchain.com/o/1c08250c-dfec-4e81-84cb-9a2fe35733e9/datasets/b0a0b9b0-62f4-403b-ac35-3a4010741091/compare?selectedSessions=c15163fc-c89b-4cc3-af08-b814dce04d06
- Raw results: [results/results_latest.csv](results/results_latest.csv)
- Computed metrics: [results/metrics.json](results/metrics.json)

## Failing examples

| Transaction | Predicted category | Expected category | Predicted merchant | Expected merchant |
|---|---|---|---|---|
| DB VERTRIEB GMBH FRANKFURT | Transport | Transport | DB Vertrieb GmbH | Deutsche Bahn |
| MIETE OKTOBER WOHNUNG HAUPTSTR 12 | Rent & Housing | Rent & Housing | MIETE OKTOBER WOHNUNG HAUPTSTR | Hausverwaltung Hauptstr 12 |
| AMAZON.DE MARKETPLACE PAYMENTS | Shopping | Shopping | AMAZON.DE MARKETPLACE | Amazon |
| NETFLIX.COM 1 8887 8877 CA | Subscriptions | **Entertainment** | NETFLIX.COM | Netflix |
| KONTOFUEHRUNGSGEBUEHR SEPTEMBER | Fees & Charges | Fees & Charges | Unknown | Bank Fee |
