import concurrent.futures, hashlib, json, os, subprocess, sys, urllib.request
from pathlib import Path
base=Path(__file__).resolve().parent
record=json.loads((base/'project/runtime/Kairos/results/environment.json').read_text())['model_verification']
dest=base/'models--Qwen--Qwen3-8B'/'snapshots'/record['revision']
dest.mkdir(parents=True,exist_ok=True)
files=json.load(urllib.request.urlopen('https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo/files?Recursive=false',timeout=10))['Data']['Files']
def fetch(row):
    name=row['file']
    item=next((f for f in files if f['Path']==name),{})
    if item.get('Sha256')==row['sha256'] and item['Size']==row['bytes']:
        url='https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/'+item['Revision']+'/'+name
    else:
        url='https://hf-mirror.com/Qwen/Qwen3-8B/resolve/'+record['revision']+'/'+name
    path=dest/name
    tmp=dest/(name+'.partial')
    subprocess.run(['curl','-fLsS','--connect-timeout','8','--max-time','900','--retry','2','-C','-','-o',str(tmp),url],check=True)
    with tmp.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=row['sha256'] or tmp.stat().st_size!=row['bytes']: raise RuntimeError('wrong bytes '+name)
    tmp.rename(path)
    print('verified',name,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(fetch,record['files']))
print('MODEL_READY',flush=True)
