"""Emit public triage counts and private ContractNLI evidence-review packets."""
import argparse
import hashlib
import json
from pathlib import Path
from cuda_summary.legal_quality import triage


def main():
    p=argparse.ArgumentParser()
    p.add_argument('raw');p.add_argument('--source',default='data/local/contract-dev.json')
    p.add_argument('--output',default='experiments/legal/quality-diagnostics.json')
    p.add_argument('--packet',default='data/local/legal-review-packet.json')
    a=p.parse_args()
    r=json.loads(Path(a.raw).read_text());source=json.loads(Path(a.source).read_text())
    if r['status']!='complete':raise ValueError('complete benchmark required')
    docs={str(d['id']):d for d in source['documents']}
    wanted=[x['id'] for x in r['input_tokens']]
    if len(set(wanted))!=len(wanted):raise ValueError('duplicate document IDs')
    output={'kind':'lexical_triage_not_factuality','raw_sha256':hashlib.sha256(Path(a.raw).read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
            'limitations':'Flags require source review; absence is not support. Word-number normalization covers zero to ninety-nine. No party, negation, exception or clause alignment. ContractNLI labels concern original documents, not generated summaries.',
            'aliases':r.get('aliases',{}),'policies':{}}
    packet={'instructions':'Assess atomic summary claims as supported, contradicted or unclear using full source and gold evidence. Record omissions separately. Do not treat NotMentioned as proof of falsity. Do not score summaries directly against ContractNLI labels.',
            'documents':[]}
    summaries={}
    for policy,rows in r['summaries'].items():
        if {x['id'] for x in rows}!=set(wanted) or len(rows)!=len(wanted):raise ValueError('summary coverage mismatch')
        summaries[policy]={x['id']:x['summary'] for x in rows}
        checks=[(x['id'],triage(docs[x['id']]['text'],x['summary'])) for x in rows]
        output['policies'][policy]={'documents':len(rows),
            'flagged_documents':sum(c['review_required'] for _,c in checks),
            'duration_mismatch_mentions':sum(c['duration_mismatch_count'] for _,c in checks),
            'numeric_anchor_mismatch_mentions':sum(c['numeric_anchor_mismatch_count'] for _,c in checks),
            'over_160_words':sum(c['over_160_words'] for _,c in checks),
            'flags_by_id':{i:{k:v for k,v in c.items() if not k.startswith('unmatched_')} for i,c in checks}}
    for key in wanted:
        d=docs[key];evidence=[]
        for annotation_set in d['annotation_sets']:
            for label,annotation in annotation_set['annotations'].items():
                spans=[{'span_index':i,'offsets':d['spans'][i],
                        'text':d['text'][d['spans'][i][0]:d['spans'][i][1]]} for i in annotation['spans']]
                evidence.append({'label':label,'hypothesis':source['labels'][label]['hypothesis'],
                                 'gold_relation':annotation['choice'],'evidence':spans})
        packet['documents'].append({'id':key,'source_url':d.get('url'),'source':d['text'],
                                    'annotations':evidence,'summaries':{k:v[key] for k,v in summaries.items()}})
    Path(a.output).write_text(json.dumps(output,indent=2)+'\n')
    Path(a.packet).parent.mkdir(parents=True,exist_ok=True)
    Path(a.packet).write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in x.items() if n!='flags_by_id'} for k,x in output['policies'].items()},indent=2))


if __name__=='__main__':main()
