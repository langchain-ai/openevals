import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset_manager import load_cases
from evaluators import schema_valid, citation_valid, status_valid

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={x['id']:x for x in load_cases()}
    def test_18_cases(self):
        self.assertEqual(len(self.cases),18)
    def test_each_case_has_evidence(self):
        for case in self.cases.values():
            self.assertTrue(case['inputs']['evidence'])
    def test_good_response_passes_deterministic_checks(self):
        c=self.cases['CS-05']
        out={'answer':'EUR 650,000','status':'supported','evidence_ids':['F1']}
        for fn in (schema_valid,citation_valid,status_valid):
            self.assertEqual(fn(c['inputs'],out,c['reference_outputs'])['score'],1)
    def test_unsupported_source_fails(self):
        c=self.cases['CS-05']
        out={'answer':'EUR 650,000','status':'supported','evidence_ids':['Z99']}
        self.assertEqual(citation_valid(c['inputs'],out,c['reference_outputs'])['score'],0)
    def test_wrong_status_fails(self):
        c=self.cases['CS-09']
        out={'answer':'Zero revenue','status':'supported','evidence_ids':['F4']}
        self.assertEqual(status_valid(c['inputs'],out,c['reference_outputs'])['score'],0)
    def test_empty_response_fails_schema(self):
        c=self.cases['CS-03']
        self.assertEqual(schema_valid(c['inputs'],{},c['reference_outputs'])['score'],0)

if __name__=='__main__':unittest.main()
