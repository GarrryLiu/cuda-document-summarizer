"""Select a deterministic length-spread subset of ContractNLI dev NDAs."""
import hashlib
import json
from pathlib import Path

source=Path('data/local/contract-dev.json')
raw=json.loads(source.read_text())
eligible=sorted([(i,d) for i,d in enumerate(raw['documents']) if 2000<=len(d['text'])<=24000],key=lambda x:len(x[1]['text']))
if len(eligible)<24: raise ValueError('not enough eligible full documents')
selected=[eligible[round(i*(len(eligible)-1)/23)] for i in range(24)]
selected.sort(key=lambda x:x[0])  # original dataset order is arrival-order baseline
rows=[{'id':str(d['id']),'text':d['text']} for _,d in selected]
Path('data/local/legal-poc.jsonl').write_text('\n'.join(json.dumps(d) for d in rows)+'\n')
manifest={'dataset':'ContractNLI','authors':'Yuta Koreeda and Christopher D. Manning (2021)',
 'source':'https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip',
 'paper':'https://aclanthology.org/2021.findings-emnlp.164/','license':'CC BY 4.0',
 'split':'dev','selection':'24 character-length quantiles among full texts with 2000–24000 characters, then restore original dev order',
 'source_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'eligible':len(eligible),'total_dev_documents':len(raw['documents']),
 'documents':[{'id':str(d['id']),'characters':len(d['text']),'source_url':d.get('url'),
               'sha256':hashlib.sha256(d['text'].encode()).hexdigest()} for _,d in selected]}
Path('experiments/legal/corpus-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Selected',len(rows),'full NDAs;',min(len(d['text']) for d in rows),max(len(d['text']) for d in rows),'characters')
