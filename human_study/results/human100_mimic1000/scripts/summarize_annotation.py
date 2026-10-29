#!/usr/bin/env python3
"""Record native automatic annotation selection after its process has exited."""
import hashlib
import json
from pathlib import Path
import re
import h5py

root=Path(__file__).resolve().parent
log=root/'logs/annotation.log'
text=log.read_text(errors='replace')
summary=re.findall(r'Exported (\d+) \(out of (\d+)\) annotated episodes?\.',text)
if len(summary)!=1:raise RuntimeError('Missing unique native annotation completion summary')
exported,processed=map(int,summary[0])
if processed!=100 or not 1<=exported<=100:raise ValueError((exported,processed))
rows=[];output_index=0
for match in re.finditer(r'Annotating episode #(\d+) \((demo_\d+)\)(.*?)(?=\nAnnotating episode #|\Z)',text,re.S):
    index,name,body=match.groups();passed='Exported the annotated episode.' in body
    rows.append({'input_iteration':int(index),'source_demo':name,'exported':passed,
                 'annotated_demo':f'demo_{output_index}' if passed else None,
                 'messages':[line.strip() for line in body.splitlines() if line.startswith('\t')]})
    output_index+=passed
assert len(rows)==100 and sum(r['exported'] for r in rows)==exported
output=root/'data/human100_annotated.hdf5'
with h5py.File(output,'r') as h:
    assert len(h['data'])==exported
    assert all(bool(g.attrs['success']) for g in h['data'].values())
    env_args=json.loads(h['data'].attrs['env_args'])
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'status':'completed','processed_count':processed,'exported_count':exported,
        'skipped_count':processed-exported,'source':str(root/'data/human100.hdf5'),
        'source_sha256':digest(root/'data/human100.hdf5'),'output':str(output),
        'output_sha256':digest(output),'log_sha256':digest(log),'output_env_args':env_args,
        'selection':'Native automatic annotation; no manually changed success or grasp thresholds.',
        'training_note':'All 100 original human demonstrations remain in the combined training data; only annotated sources seed Mimic generation.',
        'episodes':rows}
manifest=root/'annotation_manifest.json'
with manifest.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ('status','processed_count','exported_count','skipped_count')},indent=2))
