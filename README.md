# LangSmith Custom Evaluation Lab — Financial Transaction Categorization

Custom LangSmith dataset + evaluation pipeline for the "Micromanage your agents" lab.

## Domain

**Financial transaction categorization.** The input is a single raw bank-statement
transaction line the way a PSD2/open-banking feed (e.g. Enable Banking) actually returns
it: abbreviated, often in German, with card-terminal codes and reference numbers mixed in
(e.g. `REWE SAGT DANKE 8020 KARTE1 76-3//BERLIN/DE`). The expected output is a category
from a fixed 15-value taxonomy (Groceries, Dining & Restaurants, Transport, Utilities,
Rent & Housing, Subscriptions, Shopping, Healthcare, Entertainment, Travel,
Income & Salary, Transfers, Fees & Charges, Insurance, Other) plus a clean, normalized
merchant name. A good example pairs one raw transaction line with the category and
merchant an analyst would actually assign — the interesting evaluation question is
whether an LLM can both pick the right taxonomy bucket *and* clean the noisy text into a
usable merchant name.

Source documents: `data/source_transactions.txt` — 20 synthetic-but-realistic
EU-style bank export lines, each hand-labeled with ground-truth category, merchant, and a
difficulty tag (easy/medium/hard). Synthetic data was used (rather than pulling real
transactions from an existing project) to keep this repo self-contained and free of any
personal financial data.

## Dataset

- **Name:** `bank-transaction-categorization-v1`
- **LangSmith link:** https://eu.smith.langchain.com/datasets/b0a0b9b0-62f4-403b-ac35-3a4010741091
- **Size:** 20 examples
- **Structure:** each example is `{inputs: {transaction_text, amount, currency}, outputs:
  {category, merchant}, metadata: {difficulty, source, id}}`

## Experiment

- **Target model:** `gpt-4o-mini`, temperature 0
- **Evaluators:** `category_exact_match` (deterministic) + `correctness` (LLM-as-judge,
  `gpt-4o-mini`)
- **Experiment:** `txn-categorization-gpt-4o-mini-262d9e4d` — https://eu.smith.langchain.com/o/1c08250c-dfec-4e81-84cb-9a2fe35733e9/datasets/b0a0b9b0-62f4-403b-ac35-3a4010741091/compare?selectedSessions=c15163fc-c89b-4cc3-af08-b814dce04d06

See [evaluation_summary.md](evaluation_summary.md) for results and analysis.

## File map

```
data/
  source_transactions.txt   raw "documents": labeled bank-export lines
  examples.json             structured examples generated from the above
src/
  common.py                 shared config + LangSmith/OpenAI client helpers
  01_prepare_dataset.py     parses source_transactions.txt -> examples.json
  02_create_dataset.py      creates/updates the LangSmith dataset from examples.json
  03_target_function.py     @traceable target function (categorize_transaction)
  04_evaluators.py          category_exact_match + LLM-as-judge correctness evaluator
  05_run_evaluation.py      runs client.evaluate() and exports results/*.csv
  06_analyze_results.py     computes aggregate/breakdown metrics -> results/metrics.json
results/
  results_latest.csv        raw per-example outputs + scores from the experiment
  metrics.json              aggregate metrics used to write evaluation_summary.md
evaluation_summary.md       required short-form results write-up (repo root)
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` at the repo root (not committed) with:

```
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=<your project name>
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # or https://api.smith.langchain.com for US
```

Then run the pipeline in order:

```bash
python src/01_prepare_dataset.py    # data/source_transactions.txt -> data/examples.json
python src/02_create_dataset.py     # uploads examples.json to LangSmith
python src/03_target_function.py    # smoke-tests the target function on 3 examples
python src/04_evaluators.py         # smoke-tests both evaluators on a sample
python src/05_run_evaluation.py     # runs the full experiment, writes results/*.csv
python src/06_analyze_results.py    # computes metrics.json + prints the summary
```

All scripts are idempotent: re-running `02_create_dataset.py` upserts into the existing
dataset instead of creating a duplicate, and `05_run_evaluation.py` creates a new
timestamped experiment each time.

## Scope

This lab covers the core required workflow (Parts 1-5): dataset creation, a traced target
function, evaluator setup, running the LangSmith experiment, and result analysis. The
optional custom-evaluator and cost/performance A/B sections were not attempted.
