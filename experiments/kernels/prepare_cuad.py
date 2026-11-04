"""Convert the official CUAD test split into ignored local profiling inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

SHA256='f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a'
URL='https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip'


def main():
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args()
    archive=Path(a.archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Archive differs from the profiled CUAD release')
    with zipfile.ZipFile(archive) as z: data=json.loads(z.read('test.json'))
    docs=[]
    for item in data['data']:
        if len(item['paragraphs'])!=1: raise ValueError('Expected one complete contract per item')
        docs.append({'id':item['title'],'text':item['paragraphs'][0]['context']})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(d)+'\n' for d in docs))
    print(json.dumps({'documents':len(docs),'source':URL,'archive_sha256':SHA256,
                      'input_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
