"""Offline checks only. Produces illustrative failures, NOT a LangSmith run."""
import json
from pathlib import Path
from dataset_manager import load_cases
from evaluators import schema_valid, citation_valid, status_valid

BASE=Path(__file__).resolve().parent
cases={c['id']:c for c in load_cases()}
examples=[
 ('CS-05',{'answer':'FY2025 working capital = EUR 650,000; provisional.','status':'supported','evidence_ids':['F1']}),
 ('CS-08',{'answer':'Working capital is EUR +200,000.','status':'supported','evidence_ids':['F3']}),
 ('CS-15',{'answer':'The company is carbon neutral and risk free.','status':'supported','evidence_ids':['Z99']}),
 ('CS-02',{'answer':'Different entities with different registry IDs.','status':'supported','evidence_ids':['R1','R2']}),
]
results=[]
for case_id,out in examples:
 case=cases[case_id]
 scores={fn.__name__:fn(case['inputs'],out,case['reference_outputs'])['score']
         for fn in (schema_valid,citation_valid,status_valid)}
 results.append({'id':case_id,'response':out,'checks':scores,
                 'note':'CS-08 wrong numerical sign not caught by deterministic checks; live judge needed' if case_id=='CS-08' else ''})
(BASE/'artifacts'/'offline_smoke_results.json').write_text(
 json.dumps({'mode':'offline_not_langsmith','results':results},indent=2)+'\n', encoding='utf-8')
for r in results:print(r['id'],r['checks'])
print('Offline smoke proves validation wiring only; requires real LangSmith run')
