# Micromanage your agents — CompanyScope LangSmith Lab

This is a **self-contained teaching experiment**, not a production CompanyScope integration. It grades a company-research assistant on **18 fictional, human-labeled evidence-grounding cases** drawn from five synthetic source documents. It reproduces a LangSmith dataset → traced OpenAI target → evaluators → tracked experiment → report workflow.

## Setup on your Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` in VS Code. Add **both** your `OPENAI_API_KEY` and your separate **`LANGSMITH_API_KEY`**. Use the `LANGSMITH_ENDPOINT` matching the region in which you created the LangSmith key (`https://eu.api.smith.langchain.com` for EU or `https://api.smith.langchain.com` for US). The `.env` file is ignored by Git. **Do not paste keys into a chat, terminal output or GitHub.**

## Run

```bash
python dataset_manager.py --local-check
python offline_smoke.py
python -m unittest discover -s tests -v
python dataset_manager.py
python run_experiment.py --variant cautious
python analyze_results.py --input artifacts/evaluation_results_cautious.json
```

- `dataset_manager.py` uploads/validates 18 cases in `companyscope-evidence-v1`. Repeated executions check whether the dataset already has those IDs instead of blindly duplicating data.
- `run_experiment.py` runs the **real OpenAI target and OpenAI LLM judge** inside LangSmith with tracing enabled; costs depend on your accounts and model pricing. Inspect experiment feedback and traces in the LangSmith UI.
- `analyze_results.py` creates the final `evaluation_summary.md`, `lab_proof.md` and JSON analysis **from actual run results**. If a LangSmith judge row is missing, investigate before submitting.
- Optional baseline comparison: `python run_experiment.py --variant baseline`, then compare results and update `optimization_summary.md`. No A/B results are fabricated.

**Important:** Real LangSmith/LLM runs cannot be executed from this distributed file bundle because it contains **no account credentials**. Files with `offline` in their name are simulations, not live measurements.

## Submission checklist

1. Verify **18 examples** visible in your LangSmith dataset and a completed experiment with traces and feedback (LangSmith UI).
2. Check the generated real JSON in `artifacts/` and completed root `evaluation_summary.md` / `lab_proof.md`.
3. Add the **actual dataset and experiment URLs** (from your LangSmith UI) to `lab_proof.md` and optionally below. Do not invent URLs.
4. Push this repository (without `.env`), and submit its GitHub URL.

**Dataset UI URL:** paste after live upload. **Experiment UI URL:** paste after live run.

## File map

| File/folder | Purpose |
|---|---|
| `lab_proof.md` | Reviewer entry point: workflow, specific input/response, evidence, failure/limitation |
| `documents/*.md` | Five fictional company-research source documents |
| `cases.json` | 18 labeled prompts with excerpts, expected answers, reference statuses |
| `dataset_schema.md` | Domain, provenance, validation and schema |
| `dataset_manager.py` | Local checks + LangSmith dataset upload and validation |
| `run_experiment.py` | Wrapped OpenAI candidate, traced target, LangSmith evaluate run |
| `evaluators.py` | Format, citation, status and LLM correctness evaluators |
| `analyze_results.py` | Real result aggregation and short submission summaries |
| `offline_smoke.py` | Clearly labeled local demonstration of deterministic evaluators |
| `evaluation_summary.md` | One-paragraph required results summary, overwritten after live run |
| `optimization_summary.md` | Optional, honest A/B plan; update only after two real runs |
| `artifacts/` | Local smoke output, live JSON and computed analysis |
| `tests/` | Python unit tests that require no API key |
| `source_notes.md` | SDK references and limitations |
| `.env.example` | Credential template; `.env` is ignored |

Do not commit any real client information to this public teaching repository. See `source_notes.md` for implementation references.

## Live experiment

- [LangSmith evaluation: 18 scored CompanyScope examples](https://eu.smith.langchain.com/o/7c22c288-2a17-4f37-bab0-e0571ae49504/datasets/90108915-2b96-4097-a8b9-5d0d1c405e01/compare?selectedSessions=9ae98693-79b6-4d27-8002-bbadd11c6a28)
- Dataset ID: `90108915-2b96-4097-a8b9-5d0d1c405e01`
- For actual numeric results, see [`evaluation_summary.md`](evaluation_summary.md) and [`lab_proof.md`](lab_proof.md).
