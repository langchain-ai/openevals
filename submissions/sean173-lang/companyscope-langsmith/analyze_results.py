"""Summarize *real* exported LangSmith experiment data; write submission evidence."""
import argparse
import datetime as dt
import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from dataset_manager import load_cases

BASE=Path(__file__).resolve().parent
SCORES=('schema_valid','citation_valid','status_valid','llm_correctness')


def summarize(payload):
    rows=payload['rows']
    if payload.get('kind')!='real_langsmith_experiment' or not rows:
        raise ValueError('Expected nonempty export from a real LangSmith experiment')
    source_cases={c['id']:c for c in load_cases()}
    if len(rows)!=len(source_cases) or {r['case_id'] for r in rows}!=set(source_cases):
        raise ValueError('Missing/extra/duplicate case IDs; experiment incomplete')
    metric={}
    for k in SCORES:
        vals=[r.get('scores',{}).get(k) for r in rows]
        vals=[float(x) for x in vals if isinstance(x,(int,float,bool))]
        metric[k]={'count':len(vals),'mean':round(statistics.mean(vals),3) if vals else None,
                   'pass_count':sum(v==1.0 for v in vals)}
    by_category=defaultdict(list)
    for r in rows:
        sc=r.get('scores',{}).get('llm_correctness')
        if isinstance(sc,(int,float,bool)):
            by_category[source_cases[r['case_id']]['category']].append(float(sc))
    categories={k:{'n':len(v),'mean':round(statistics.mean(v),3)} for k,v in sorted(by_category.items())}
    times=[]
    for r in rows:
        try:
            start=dt.datetime.fromisoformat(r['run_start_time'])
            end=dt.datetime.fromisoformat(r['run_end_time'])
            times.append((end-start).total_seconds())
        except (TypeError,ValueError):pass
    dollars=[r['run_total_cost'] for r in rows if isinstance(r.get('run_total_cost'),(float,int))]
    tokens=[r['run_total_tokens'] for r in rows if isinstance(r.get('run_total_tokens'),(float,int))]
    failures=[{'case_id':r['case_id'], 'category':source_cases[r['case_id']]['category'],
               'reason':str(r.get('judge_comments',{}).get('llm_correctness',''))[:280]}
              for r in rows if r.get('scores',{}).get('llm_correctness')==0]
    return {'row_count':len(rows),'metrics':metric,'categories':categories,
            'latency_seconds':{'count':len(times),'mean':round(statistics.mean(times),3) if times else None,
                               'median':round(statistics.median(times),3) if times else None},
            'total_target_tokens_observed':sum(tokens) if tokens else None,
            'total_target_cost_usd_observed':round(sum(dollars),8) if dollars else None,
            'failures':failures,
            'model':payload['model'],'variant':payload['variant'],
            'experiment_name':payload.get('experiment_name',''),
            'dataset':payload['dataset']}


def fmt(value):
    return 'not available' if value is None else str(value)


def write_report(s, source):
    judged=s['metrics']['llm_correctness']
    total=s['row_count']
    failed=s['failures']
    cattext=', '.join(f'{k}: {v["mean"]:.0%} (n={v["n"]})' for k,v in s['categories'].items())
    worst=', '.join(f['case_id'] for f in failed[:3]) if failed else 'none in this small dataset'
    p=(f"We evaluated **{total} synthetic CompanyScope evidence-grounding tasks** with the "
       f"traced `{s['model']}` target (`{s['variant']}` instructions) in the LangSmith "
       f"experiment `{s['experiment_name'] or 'name unavailable in export'}`. "
       f"A reference-aware LLM judge scored **{judged['pass_count']}/{judged['count']}** "
       f"scorable responses as correct ({fmt(judged['mean'])} mean), while deterministic "
       f"citation validation passed **{s['metrics']['citation_valid']['pass_count']}/{s['metrics']['citation_valid']['count']}**. "
       f"The observed failed cases included **{worst}**; category variation should be "
       f"interpreted cautiously given tiny cell sizes. Mean target latency was "
       f"**{fmt(s['latency_seconds']['mean'])} seconds**; observed target token use was "
       f"**{fmt(s['total_target_tokens_observed'])}** and provider-reported target cost "
       f"**{fmt(s['total_target_cost_usd_observed'])} USD** (missing values are not zero; "
       f"judge calls may not be included). This synthetic dataset and automated judging "
       f"do **not** prove correctness on live company records; next, manually review "
       f"failed cases, strengthen source checks and rerun on a blinded human-reviewed sample.")
    lines=['# Evaluation summary — real LangSmith run','',p,'',
           '| Metric | Observed |','|---|---:|',
           f"| Cases | {total} |",
           f"| LLM correctness judged | {judged['pass_count']} / {judged['count']} |",
           f"| Schema valid | {s['metrics']['schema_valid']['pass_count']} / {s['metrics']['schema_valid']['count']} |",
           f"| Citation valid | {s['metrics']['citation_valid']['pass_count']} / {s['metrics']['citation_valid']['count']} |",
           f"| Status correct | {s['metrics']['status_valid']['pass_count']} / {s['metrics']['status_valid']['count']} |",
           f"| Mean target latency (s) | {fmt(s['latency_seconds']['mean'])} |",
           f"| Target tokens reported | {fmt(s['total_target_tokens_observed'])} |",
           f"| Target cost reported (USD) | {fmt(s['total_target_cost_usd_observed'])} |",'',
           '**By category:** '+(cattext or 'No judge data available.'),'',
           f'**Source export:** `{source}`. See `artifacts/analysis_summary.json` for detailed computed metrics.','',
           '**Evidence limitation:** these are invented company records; independently verify facts and have a human adjudicate borderline scores before deployment.','']
    (BASE/'evaluation_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    proof=f"""# Lab proof — CompanyScope LangSmith evaluation

## Code and workflow
- Data: [`cases.json`](cases.json) + [`documents/`](documents/) (18 independent synthetic examples).
- Upload/verify: [`dataset_manager.py`](dataset_manager.py).
- Real traced target, deterministic + LLM correctness judges: [`run_experiment.py`](run_experiment.py), [`evaluators.py`](evaluators.py).
- Analysis: [`analyze_results.py`](analyze_results.py), [`evaluation_summary.md`](evaluation_summary.md).

## One real run: input → evidence → result
- Input example: **CS-05**, "Calculate Aurora Vector GmbH's FY2025 working capital" using F1: current assets EUR 1,750,000 and current liabilities EUR 1,100,000.
- Expected calculation: **EUR 650,000**, with source F1, and provisional FY2025 caveat.
- Actual export: [`{source}`]({source}), containing this case's real target output, judge explanation and metric scores (inspect row `case_id=CS-05`).
- Dataset: `{s['dataset']}`; experiment: `{s['experiment_name'] or 'open LangSmith UI to identify experiment'}`; model: `{s['model']}`.
- Final outcome: {judged['pass_count']} of {judged['count']} cases passed LLM correctness (mean {fmt(judged['mean'])}); citations passed {s['metrics']['citation_valid']['pass_count']} / {s['metrics']['citation_valid']['count']}.

## Limitation / failure
- Example hard failure modes: invented financial amounts, entity confusion, malicious instructions in A1/A2 and missing information. Real failed case IDs: **{worst}**.
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
"""
    (BASE/'lab_proof.md').write_text(proof,encoding='utf-8')
    (BASE/'artifacts'/'analysis_summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    args=p.parse_args()
    source=Path(args.input)
    if not source.is_absolute():source=BASE/source
    payload=json.loads(source.read_text(encoding='utf-8'))
    s=summarize(payload)
    rel=str(source.relative_to(BASE)) if source.is_relative_to(BASE) else str(source)
    write_report(s,rel)
    print(f"Created evaluation_summary.md, lab_proof.md, artifacts/analysis_summary.json from {rel}")
    print(f"LLM correctness: {s['metrics']['llm_correctness']['pass_count']} / {s['metrics']['llm_correctness']['count']}")
    if s['metrics']['llm_correctness']['count']!=s['row_count']:
        raise RuntimeError('Some rows lack LLM judge scores; review LangSmith errors before submission')

if __name__=='__main__':main()
