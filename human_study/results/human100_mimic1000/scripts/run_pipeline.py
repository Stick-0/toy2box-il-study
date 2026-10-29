#!/usr/bin/env python3
"""Run the requested generation, fixed training, and recorded evaluation locally."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

import h5py

ROOT=Path(__file__).resolve().parent
LAB=Path('/home/jm/robot_sim/IsaacLab_RS')
REPO=Path('/home/jm/robot_sim/toy2box-il-study')
TASK='Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0'
MIMIC_TASK='Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0'
LAUNCH=[str(LAB/'isaaclab.sh'),'-p','-u']
ACTIVE=[]
STATUS={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'stages':{}}


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def status():save(ROOT/'pipeline_status.json',STATUS)


def check_sources():
    for path,digest in json.loads((ROOT/'native_source_hashes.json').read_text()).items():
        if sha(path)!=digest:raise RuntimeError('Native source changed: '+path)


def run_group(stage,jobs):
    """Jobs are independent; preserve their logs and fail before dependent work."""
    STATUS['current_stage']=stage
    row={'status':'running','started_at':now(),'jobs':[]}
    STATUS['stages'][stage]=row;status()
    streams=[];processes=[]
    try:
        for label,command,logfile in jobs:
            stream=logfile.open('x',buffering=1);streams.append(stream)
            proc=subprocess.Popen(command,cwd=LAB,stdout=stream,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,start_new_session=True)
            processes.append(proc);ACTIVE.append(proc)
            row['jobs'].append({'label':label,'command':command,'log':str(logfile),'pid':proc.pid})
            status()
            print(f'[{now()}] {stage}: started {label} pid={proc.pid}',flush=True)
        while any(p.poll() is None for p in processes):
            failed=[p for p in processes if p.poll() not in (None,0)]
            if failed:raise RuntimeError(f'{stage} subprocess failed: {[p.returncode for p in failed]}')
            time.sleep(2)
        if any(p.returncode!=0 for p in processes):raise RuntimeError(f'{stage} returned nonzero')
        row.update(status='completed',finished_at=now(),returncodes=[p.returncode for p in processes]);status()
        print(f'[{now()}] {stage}: completed',flush=True)
    except BaseException as exc:
        row.update(status='failed',finished_at=now(),error=f'{type(exc).__name__}: {exc}')
        status()
        raise
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
            ACTIVE.remove(p)
        for stream in streams:stream.close()
        row['returncodes']=[p.returncode for p in processes]
        status()


def main():
    if (ROOT/'pipeline_status.json').exists():
        raise FileExistsError('Pipeline status exists; inspect previous run before resuming')
    annotation=json.loads((ROOT/'annotation_manifest.json').read_text())
    assert annotation['status']=='completed' and annotation['processed_count']==100
    annotated=ROOT/'data/human100_annotated.hdf5'
    assert sha(annotated)==annotation['output_sha256']
    human=ROOT/'data/human100.hdf5'
    plan=json.loads((ROOT/'run_plan.json').read_text())
    assert sha(human)==plan['human_sha256']
    check_sources();status()
    mimic=ROOT/'data/human100_mimic1000_raw.hdf5'
    combined=ROOT/'data/human100_plus_mimic1000.hdf5'
    for path in [mimic,combined,ROOT/'dataset_manifest.json']:
        if path.exists():raise FileExistsError(path)
    run_group('generation',[(
        'mimic1000',LAUNCH+[str(LAB/'scripts/imitation_learning/isaaclab_mimic/generate_dataset.py'),
        '--task',MIMIC_TASK,'--headless','--num_envs','8','--generation_num_trials','1000',
        '--input_file',str(annotated),'--output_file',str(mimic)],ROOT/'logs/generation.log')])
    with h5py.File(mimic,'r') as h:
        generated=len(h['data']);successful=sum(bool(g.attrs['success']) for g in h['data'].values())
    if successful<1000:raise RuntimeError(f'Only {successful} successful generated demonstrations')
    matches=re.findall(r'(\d+)/(\d+) \([0-9.]+%\) successful demos generated',
                       (ROOT/'logs/generation.log').read_text(errors='replace'))
    save(ROOT/'generation_summary.json',{
        'status':'completed','num_envs':8,'target_successes':1000,'successful_file_count':successful,
        'raw_file_episode_count':generated,'last_native_counts':list(map(int,matches[-1])) if matches else None,
        'input_annotations':annotation['exported_count'],'raw_file':str(mimic),'raw_sha256':sha(mimic),
        'selection':'First 1000 successful records in numeric order; failed/overflow records excluded from training.'})
    run_group('prepare_training_data',[(
        'combine1100',LAUNCH+[str(REPO/'human_study/scripts/prepare_mimic_training.py'),
        '--human',str(human),'--mimic',str(mimic),'--output',str(combined),
        '--manifest',str(ROOT/'dataset_manifest.json'),'--limit','1000'],ROOT/'logs/prepare.log')])
    manifest=json.loads((ROOT/'dataset_manifest.json').read_text())
    assert manifest['output']['human_count']==100 and manifest['output']['mimic_count']==1000
    assert manifest['output']['demo_count']==1100
    jobs=[]
    for seed in (101,111):
        command=LAUNCH+[str(ROOT/'train_with_local_config.py'),'--seed',str(seed),
            '--config',str(LAB/'source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/agents/robomimic/bc_rnn_low_dim.json'),
            '--baseline-config',str(REPO/f'human_study/results/mimic500_20261028/configs/baseline_seed{seed}.json'),
            '--native-source',str(LAB/'scripts/imitation_learning/robomimic/train.py'),
            '--lab-dir',str(LAB),'--dataset',str(combined),'--name',f'human100_mimic1000_seed{seed}',
            '--log-dir',str(ROOT/'native_training'),'--output-dir',str(ROOT/'training'/f'seed{seed}')]
        subprocess.run(command+['--validate-only'],cwd=LAB,check=True)
        jobs.append((f'seed{seed}',command,ROOT/'logs'/f'train_seed{seed}.log'))
    run_group('training',jobs)
    models=[]
    for seed in (101,111):
        trained=json.loads((ROOT/'training'/f'seed{seed}'/'summary.json').read_text())
        assert trained['status']=='completed' and trained['seed']==seed
        source=Path(trained['checkpoint']);destination=ROOT/'models'/f'human100_mimic1000_seed{seed}.pth'
        if destination.exists():raise FileExistsError(destination)
        assert sha(source)==trained['checkpoint_sha256']
        shutil.copy2(source,destination)
        assert sha(destination)==trained['checkpoint_sha256']
        models.append({'training_seed':seed,'path':str(destination),'sha256':sha(destination),'epoch':2000})
    save(ROOT/'models/manifest.json',{'models':models,'selection':'Fixed final epoch 2000'})
    jobs=[];direct_commands=[]
    for model in models:
        seed=model['training_seed'];folder=ROOT/'evaluation'/f'seed{seed}'
        native_args=['--task',TASK,'--checkpoint',model['path'],'--seed','42','--num_rollouts','100',
                     '--horizon','800','--headless','--video','--video_length','1000','--video_folder',str(folder/'videos')]
        command=LAUNCH+[str(ROOT/'evaluate_with_telemetry.py'),'--output-dir',str(folder),
            '--training-seed',str(seed),'--native-play',str(LAB/'scripts/imitation_learning/robomimic/play.py'),*native_args]
        jobs.append((f'seed{seed}',command,ROOT/'logs'/f'evaluation_seed{seed}.log'))
        direct_commands.append(shlex.join(LAUNCH+[str(LAB/'scripts/imitation_learning/robomimic/play.py'),*native_args]))
    (ROOT/'evaluation_commands.txt').write_text('\n\n'.join(direct_commands)+'\n')
    run_group('evaluation',jobs)
    check_sources()
    run_group('summarize',[(
        'results',LAUNCH+[str(ROOT/'summarize_run.py')],ROOT/'logs/summarize.log')])
    STATUS.update(status='completed',finished_at=now());status()
    print(f'[{now()}] All requested stages completed. Results: {ROOT}/results.txt',flush=True)


if __name__=='__main__':
    if (ROOT/'pipeline_status.json').exists():
        raise SystemExit('Pipeline status exists; inspect previous run before resuming')
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:main()
    except BaseException as exc:
        STATUS.update(status='failed',finished_at=now(),error=f'{type(exc).__name__}: {exc}')
        status()
        raise
