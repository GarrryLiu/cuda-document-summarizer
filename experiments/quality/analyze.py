"""Aggregate the pilot and strip generated prose from the public report."""
import argparse
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',default='experiments/quality/aggregate-results.json');args=p.parse_args()
r=json.loads(Path(args.input).read_text()); rows=[]
for c in r['results']:
    docs=c['documents']; valid=[d for d in docs if d['diagnostics']['numeric_coverage'] is not None]
    refs=[d for d in docs if d['diagnostics']['source_numbers']]
    quotes=sum(d['diagnostics']['evidence_quotes'] for d in docs)
    seconds=sum(t['seconds'] for t in c['batch_timings'])
    row={'prompt':c['prompt'],'cap':c['max_new_tokens'],'documents':len(docs),
         'hit_cap':sum(d['hit_output_cap'] for d in docs),
         'four_sections':sum(d['diagnostics']['section_count']==4 for d in docs),
         'valid_schema':sum(d['diagnostics']['schema_valid'] is True for d in docs) if c['prompt']=='evidence_json' else None,
         'numeric_scorable_documents':len(valid),
         'numeric_coverage_valid_mean':statistics.mean(d['diagnostics']['numeric_coverage'] for d in valid) if valid else None,
         'numeric_coverage_invalid_zero':sum(d['diagnostics']['numeric_coverage'] or 0 for d in refs)/len(refs) if refs else None,
         'exact_quotes':sum(d['diagnostics']['exact_evidence_quotes'] for d in docs), 'quotes':quotes,
         'documents_with_novel_numeric_mentions':sum(bool(d['diagnostics']['novel_numbers']) for d in docs),
         'generation_seconds':seconds,'documents_per_second':len(docs)/seconds if seconds else None,
         'generated_tokens':sum(d['generated_tokens'] for d in docs),
         'peak_allocated_bytes':c.get('peak_allocated_bytes'),
         'per_document':[{k:d[k] for k in ('id','input_tokens','generated_tokens','ended_with_eos','hit_output_cap','diagnostics')} for d in docs]}
    rows.append(row)
    print(c['prompt'],c['max_new_tokens'],'cap hits',row['hit_cap'],'sections',row['four_sections'],
          'schema',row['valid_schema'],'numeric',row['numeric_coverage_valid_mean'],'seconds',round(seconds,2))
out={k:v for k,v in r.items() if k!='results'}
out['results']=rows
out['limitations']='Abstract-only exploratory pilot, one run per condition. Numeric overlap is not fact coverage; exact evidence is not entailment. Structured parse failures are unscorable; invalid-zero metric is an operational proxy. Raw predictions retained locally, not published.'
Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
