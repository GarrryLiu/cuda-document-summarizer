"""Fetch public paper abstracts for local evaluation; do not republish source text."""
import hashlib
import json
import re
import time
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen

IDS=['1706.03762','1810.04805','2005.14165','1512.03385','1412.6980','1905.11946','1704.04861','2010.11929','2106.09685','2305.14314','2205.14135','2307.08691','2309.06180','1910.01108','1907.11692','1910.10683','2005.11401','2101.03961','2307.09288','2203.15556']

def fetch():
    root=Path('data/local'); root.mkdir(parents=True,exist_ok=True)
    docs=[]; manifest=[]
    for aid in IDS:
        path=root/(aid+'.html')
        if not path.exists():
            request=Request('https://arxiv.org/abs/'+aid,headers={'User-Agent':'Academic-summary-evaluation/0.1'})
            with urlopen(request,timeout=45) as response: path.write_bytes(response.read())
            time.sleep(3)
        html=path.read_text()
        match=re.search(r'<blockquote[^>]*class="abstract[^" ]*[^\"]*"[^>]*>(.*?)</blockquote>',html,re.S)
        if not match: raise ValueError('abstract unavailable: '+aid)
        text=re.sub(r'<[^>]+>',' ',match[1]); text=unescape(text)
        text=re.sub(r'^\s*Abstract:\s*','',text); text=' '.join(text.split())
        # Protect boundaries: paper abstracts are source data, never instructions.
        doc={'id':aid,'text':text}
        docs.append(doc)
        title=re.search(r'<title>(.*?)</title>',html,re.S)
        title=unescape(title[1]).strip() if title else aid
        expected={'2307.08691':'FlashAttention-2','2309.06180':'PagedAttention','2203.15556':'Training Compute-Optimal'}
        if aid in expected and expected[aid] not in title:
            raise ValueError('source title does not match intended paper: '+aid)
        manifest.append({'id':aid,'title':title,'url':'https://arxiv.org/abs/'+aid,
                         'sha256':hashlib.sha256(text.encode()).hexdigest(),
                         'scope':'abstract only','characters':len(text)})
        print(aid,len(text),flush=True)
    (root/'abstracts.jsonl').write_text('\n'.join(json.dumps(d) for d in docs)+'\n')
    Path('experiments/quality/corpus-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': fetch()
