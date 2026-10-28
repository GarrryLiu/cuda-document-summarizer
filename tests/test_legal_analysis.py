import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT=Path(__file__).resolve().parents[1]/'experiments/legal/analyze.py'


class AnalysisTests(unittest.TestCase):
    def test_quality_packet_is_private_and_source_identity_is_checked(self):
        row={'id':'d','text':'Private source text'}
        sha=hashlib.sha256((json.dumps(row)+'\n').encode()).hexdigest()
        raw={'status':'complete','input_sha256':sha,'input_tokens':[{'id':'d'}],
             'summaries':{'arrival_4':[{'id':'d','summary':'PRIVATE GENERATED TEXT'}]}}
        source={'labels':{},'documents':[{**row,'spans':[],'annotation_sets':[]}]}
        env={**os.environ,'PYTHONPATH':str(SCRIPT.parents[2]/'src')}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);r=path/'raw.json';s=path/'source.json';o=path/'public.json';p=path/'private.json'
            r.write_text(json.dumps(raw));s.write_text(json.dumps(source))
            cmd=[sys.executable,str(SCRIPT.with_name('quality.py')),str(r),'--source',str(s),'--output',str(o),'--packet',str(p)]
            subprocess.run(cmd,check=True,capture_output=True,env=env)
            self.assertNotIn('PRIVATE GENERATED TEXT',o.read_text())
            self.assertIn('PRIVATE GENERATED TEXT',p.read_text())
            source['documents'][0]['text']='Wrong source version';s.write_text(json.dumps(source))
            failed=subprocess.run(cmd,capture_output=True,env=env)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn(b'source text does not reproduce',failed.stderr)

    def test_alias_metrics_frontier_and_public_redaction(self):
        base={'generated_tokens_per_second':2,'peak_allocated_bytes':2**30,
              'generated_tokens':10,'cap_hits':0,'outputs_equal_to_arrival_first':1}
        r={'status':'complete','arguments':{'input':'/private/school/data.jsonl','output':'/private/school/raw.json'},
           'plans':{'arrival_4':{},'length_8':{},'budget_49152':{}},
           'aliases':{'budget_49152':'length_8'},
           'runs':[{**base,'strategy':'arrival_4','seconds':10,'documents_per_second':.1},
                   {**base,'strategy':'length_8','seconds':5,'documents_per_second':.2}],
           'summaries':{'arrival_4':[{'id':'a','summary':'PRIVATE GENERATED TEXT'}]}}
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/'raw.json';out=Path(temp)/'out.json'
            raw.write_text(json.dumps(r))
            subprocess.run([sys.executable,str(SCRIPT),str(raw),'--output',str(out)],check=True,capture_output=True)
            text=out.read_text();result=json.loads(text)
        self.assertNotIn('PRIVATE GENERATED TEXT',text)
        self.assertNotIn('/private/school',text)
        self.assertEqual(result['aggregate']['budget_49152']['measurement_source'],'length_8')
        self.assertEqual(result['aggregate']['budget_49152']['throughput_speedup_vs_arrival4'],2)
        self.assertEqual(result['observed_frontier'],['length_8','budget_49152'])

    def test_partial_report_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/'raw.json';out=Path(temp)/'out.json'
            raw.write_text('{"status":"running"}')
            run=subprocess.run([sys.executable,str(SCRIPT),str(raw),'--output',str(out)],capture_output=True)
            self.assertNotEqual(run.returncode,0)
            self.assertFalse(out.exists())
