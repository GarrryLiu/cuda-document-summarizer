import json
import unittest
from cuda_summary.quality import diagnostics,messages,numbers

class QualityTests(unittest.TestCase):
    def test_baseline_sections_and_numbers(self):
        r=diagnostics('Improved accuracy to 90% on 20 tasks.','Objectives:\nTest\nMethods:\nUse 20 tasks\nQuantitative findings:\n90% accuracy\nLimitations:\nNot reported.','baseline')
        self.assertEqual(r['section_count'],4);self.assertEqual(r['numeric_coverage'],1)
    def test_json_evidence_not_credited_as_claim(self):
        obj={'objectives':'Test','methods':'Compare','quantitative_findings':[{'claim':'Better accuracy','evidence':'90% accuracy'}],'limitations':'Not reported'}
        r=diagnostics('90% accuracy',json.dumps(obj),'evidence_json')
        self.assertTrue(r['schema_valid']);self.assertEqual(r['exact_evidence_quotes'],1)
        self.assertEqual(r['numeric_coverage'],0)
    def test_malformed_is_unscorable(self):
        r=diagnostics('100 tasks','{"objectives":','evidence_json')
        self.assertFalse(r['json_valid']);self.assertIsNone(r['numeric_coverage'])
    def test_bad_quote_and_number(self):
        obj={'objectives':'Test','methods':'Compare','quantitative_findings':[{'claim':'95% accuracy','evidence':'invented quote'}],'limitations':'Not reported'}
        r=diagnostics('90% accuracy',json.dumps(obj),'evidence_json')
        self.assertEqual(r['exact_evidence_quotes'],0);self.assertEqual(r['novel_numbers'],['95%'])
    def test_prompt_variant_and_list_markers(self):
        self.assertEqual(numbers('1. Results: 90% on 20 tasks.'),{'90%','20'})
        with self.assertRaises(ValueError): messages('text','unknown')
