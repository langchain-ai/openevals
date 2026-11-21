# Lab proof — CompanyScope LangSmith evaluation

## Code and workflow
- Data: [`cases.json`](cases.json) + [`documents/`](documents/) (18 independent synthetic examples).
- Upload/verify: [`dataset_manager.py`](dataset_manager.py).
- Real traced target, deterministic + LLM correctness judges: [`run_experiment.py`](run_experiment.py), [`evaluators.py`](evaluators.py).
- Analysis: [`analyze_results.py`](analyze_results.py), [`evaluation_summary.md`](evaluation_summary.md).

## One real run: input → evidence → result
- Input example: **CS-05**, "Calculate Aurora Vector GmbH's FY2025 working capital" using F1: current assets EUR 1,750,000 and current liabilities EUR 1,100,000.
- Expected calculation: **EUR 650,000**, with source F1, and provisional FY2025 caveat.
- Actual export: [`artifacts/evaluation_results_cautious.json`](artifacts/evaluation_results_cautious.json), containing this case's real target output, judge explanation and metric scores (inspect row `case_id=CS-05`).
- Dataset: `companyscope-evidence-v1`; experiment: `companyscope-cautious-gpt-4o-mini-9a8fb168`; model: `gpt-4o-mini`.
- Final outcome: 17 of 18 cases passed LLM correctness (mean 0.944); citations passed 0 / 18.

## Limitation / failure
- Example hard failure modes: invented financial amounts, entity confusion, malicious instructions in A1/A2 and missing information. Real failed case IDs: **CS-15**.
- The LLM judge may be wrong or biased; only 18 invented inputs were studied and source verification was not performed. This is NOT an audit of real companies.
- Inspect the dataset and experiment in your LangSmith account, then paste their actual URL(s) here after UI review. No links are invented.

## Reproduce
```bash
python dataset_manager.py --local-check
python dataset_manager.py
python run_experiment.py --variant cautious
python analyze_results.py --input artifacts/evaluation_results_cautious.json
python -m unittest discover -s tests -v
```

## Actual LangSmith evidence links

- **Experiment (18 scored cases):** [https://eu.smith.langchain.com/o/7c22c288-2a17-4f37-bab0-e0571ae49504/datasets/90108915-2b96-4097-a8b9-5d0d1c405e01/compare?selectedSessions=9ae98693-79b6-4d27-8002-bbadd11c6a28](https://eu.smith.langchain.com/o/7c22c288-2a17-4f37-bab0-e0571ae49504/datasets/90108915-2b96-4097-a8b9-5d0d1c405e01/compare?selectedSessions=9ae98693-79b6-4d27-8002-bbadd11c6a28)
- **Dataset:** `companyscope-evidence-v1` (LangSmith dataset ID `90108915-2b96-4097-a8b9-5d0d1c405e01`).
- **Recorded outcome:** see `artifacts/evaluation_results_cautious.json` and `artifacts/analysis_summary.json` for the current scores, case-level evidence and reviewer comments.
