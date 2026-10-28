"""Summarize real legal-scheduling runs, omitting contract/summary text from Git."""
import argparse
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',default='experiments/legal/aggregate-results.json');a=p.parse_args()
r=json.loads(Path(a.input).read_text())
if r['status']!='complete':raise ValueError('incomplete run')
out={k:v for k,v in r.items() if k!='summaries'}
out['arguments']={**r['arguments'], 'input':Path(r['arguments']['input']).name,
                  'output':Path(r['arguments']['output']).name}
out['summary_hashes']={k:{d['id']:__import__('hashlib').sha256(d['summary'].encode()).hexdigest() for d in values} for k,values in r['summaries'].items()}
metrics={}
for name in r['plans']:
    measured=r.get('aliases',{}).get(name,name)
    runs=[x for x in r['runs'] if x['strategy']==measured]
    metrics[name]={'median_seconds':statistics.median(x['seconds'] for x in runs),
      'measurement_source':measured,
      'median_docs_per_second':statistics.median(x['documents_per_second'] for x in runs),
      'median_generated_tokens_per_second':statistics.median(x['generated_tokens_per_second'] for x in runs),
      'max_peak_allocated_gib':max(x['peak_allocated_bytes'] for x in runs)/2**30,
      'output_token_totals':[x['generated_tokens'] for x in runs],
      'cap_hits':[x['cap_hits'] for x in runs],
      'exact_output_matches_to_arrival_first':[x['outputs_equal_to_arrival_first'] for x in runs]}
reference=metrics['arrival_4']['median_seconds']
for name,m in metrics.items():
    m['throughput_speedup_vs_arrival4']=reference/m['median_seconds']
    print(name,json.dumps(m))
out['aggregate']=metrics
out['limitations']='Offline length-spread dev subset of 24 NDAs. Token slots are a proxy, not a VRAM guarantee. Batch-shape changes can change greedy outputs. No online latency or validated legal factuality result. Fixed8 control separates increased concurrency from token-budget effects.'
Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
