#!/usr/bin/env python3
"""Record the course's fixed 100-rollout self-evaluation and its telemetry.

The native main/rollout, model, task, actions and termination logic are unchanged.
The previously validated observer records terminal states before auto-reset.
Outputs must be a new local directory; existing evaluations are never replaced.
"""
import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

TASK = 'Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0'
ROLLOUTS, EVAL_SEED, HORIZON = 100, 42, 800
TRIAL = re.compile(r'^\[INFO\] Trial (\d+): (True|False)$')
COUNTS = re.compile(r'^Successful trials: (\d+), out of (\d+) trials$')
RATE = re.compile(r'^Success rate: ([0-9.eE+-]+)$')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_settings(args):
    expected = {'task': TASK, 'seed': EVAL_SEED, 'num_rollouts': ROLLOUTS, 'horizon': HORIZON}
    for name, required in expected.items():
        if getattr(args, name) != required:
            raise ValueError(f'--{name} must be {required!r}, got {getattr(args, name)!r}')
    if not args.video or args.video_length < HORIZON + 1:
        raise ValueError('--video and --video_length >= 801 are required to retain full rollouts')
    if args.norm_factor_min is not None or args.norm_factor_max is not None:
        raise ValueError('Action normalization overrides are not part of this evaluation')


def preflight(native_args):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_rollouts', type=int, required=True)
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video_length', type=int, default=800)
    parser.add_argument('--video_folder', type=Path, required=True)
    parser.add_argument('--norm_factor_min', type=float)
    parser.add_argument('--norm_factor_max', type=float)
    args, _ = parser.parse_known_args(native_args)
    validate_settings(args)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.video_folder = args.video_folder.resolve()
    return args


def source_paths(native):
    lab = native.parents[3]
    relative = [
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/place_toy2box_rmp_rel_env_cfg.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/__init__.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/mdp/observations.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/mdp/terminations.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/mdp/franka_stack_events.py',
        'source/isaaclab/isaaclab/envs/manager_based_rl_env.py',
    ]
    return [native, Path(__file__).resolve(), *(lab / p for p in relative)]


class NativeResults:
    """Inspect only native print calls; preserve their text and behavior."""
    def __init__(self):
        self.trials, self.counts, self.rates = [], [], []

    def feed(self, line):
        line = line.strip()
        match = TRIAL.fullmatch(line)
        if match:
            self.trials.append({'trial': int(match[1]), 'success': match[2] == 'True'})
        match = COUNTS.fullmatch(line)
        if match:
            self.counts.append([int(match[1]), int(match[2])])
        match = RATE.fullmatch(line)
        if match:
            self.rates.append(float(match[1]))

    def print(self, *values, **kwargs):
        builtins.print(*values, **kwargs)
        if kwargs.get('file') not in (None, sys.stdout):
            return
        for line in kwargs.get('sep', ' ').join(map(str, values)).splitlines():
            self.feed(line)

    def validate(self, observed_records):
        expected = [{'trial': r['trial'], 'success': r['success']} for r in observed_records]
        if [r['trial'] for r in expected] != list(range(ROLLOUTS)):
            raise ValueError('Telemetry trials are not exactly 0..99')
        if self.trials != expected:
            raise ValueError('Native printed trial results disagree with telemetry records')
        count = sum(r['success'] for r in expected)
        if self.counts != [[count, ROLLOUTS]]:
            raise ValueError(f'Native count summary disagrees with telemetry: {self.counts}')
        if len(self.rates) != 1 or not math.isclose(self.rates[0], count / ROLLOUTS, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'Native success rate disagrees with telemetry: {self.rates}')

class Observer:
    def __init__(self, env):
        self.env=env
        self.unwrapped=env.unwrapped
        self.states=[]; self.actions=[]; self.observations={}; self.terminations=[]
        self.in_step=False; self.terminal_snapshot=None; self.steps=0
        self.robot=self.unwrapped.scene['robot']
        self.joints,_=self.robot.find_joints(self.unwrapped.cfg.gripper_joint_names)
        self.original_reset=self.unwrapped._reset_idx
        def observed_reset(ids):
            if self.in_step:
                self.terminal_snapshot=self.capture()
            return self.original_reset(ids)
        self.unwrapped._reset_idx=observed_reset

    def capture(self):
        scene=self.unwrapped.scene
        chunks=[scene['toy_truck'].data.root_state_w[0],scene['box'].data.root_state_w[0],
                scene['ee_frame'].data.target_pos_w[0,0],scene['ee_frame'].data.target_quat_w[0,0],
                self.robot.data.joint_pos[0,self.joints],scene['contact_grasp'].data.net_forces_w[0].reshape(-1),
                self.robot.data.root_pos_w[0],self.robot.data.root_quat_w[0]]
        state=torch.cat(chunks).detach().cpu().numpy().copy()
        assert state.shape==(48,),state.shape
        return state

    def save_obs(self, obs):
        for k,v in obs['policy'].items():
            self.observations.setdefault(k,[]).append(v[0].detach().cpu().numpy().copy())

    def reset(self,*a,**kw):
        result=self.env.reset(*a,**kw)
        self.states.append(self.capture());self.save_obs(result[0])
        return result

    def step(self, action):
        self.actions.append(action[0].detach().cpu().numpy().copy())
        self.in_step=True;self.terminal_snapshot=None
        try: result=self.env.step(action)
        finally:self.in_step=False
        auto_reset=self.terminal_snapshot is not None
        self.states.append(self.terminal_snapshot if auto_reset else self.capture())
        self.save_obs(result[0])
        self.terminations.append([bool(result[2][0]),bool(result[3][0]),auto_reset])
        self.steps+=1
        if self.steps%200==0:print(f'[EVAL] seed={own.training_seed} trial={len(records)} step={self.steps}',flush=True)
        return result

    def close_observer(self):
        self.unwrapped._reset_idx=self.original_reset

    def __getattr__(self,name): return getattr(self.env,name)

def observed_rollout(policy, env, success_term, horizon, device):
    observer=Observer(env); start=time.monotonic()
    try:
        success,traj=native_rollout(policy,observer,success_term,horizon,device)
        trial=len(records)
        arrays={'states':np.asarray(observer.states),'actions':np.asarray(observer.actions),
                'termination_flags':np.asarray(observer.terminations,dtype=bool),
                **{'obs_'+k:np.asarray(v) for k,v in observer.observations.items()}}
        assert len(arrays['states'])==len(arrays['actions'])+1
        assert all(np.isfinite(v).all() for v in arrays.values())
        filename=f'trial_{trial:03d}.npz'
        np.savez_compressed(out/filename,**arrays)
        record={'training_seed':own.training_seed,'trial':trial,'success':bool(success),
                'steps':observer.steps,'step_dt':float(env.unwrapped.step_dt),
                'elapsed_seconds':time.monotonic()-start,'trajectory':filename,
                'trajectory_sha256':sha(out/filename),'auto_reset':bool(arrays['termination_flags'][:,2].any())}
        records.append(record)
        with (out/'trials.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
            f.flush()
            os.fsync(f.fileno())
        print('[EVAL RESULT] '+json.dumps(record),flush=True)
        return success,traj
    finally: observer.close_observer()


def main():
    global own, out, records, native_rollout, np, torch
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-seed', type=int, required=True, choices=[101, 111])
    parser.add_argument('--native-play', type=Path, default=Path('/home/jm/robot_sim/IsaacLab_RS/scripts/imitation_learning/robomimic/play.py'))
    own, native_args = parser.parse_known_args()
    original_command = list(sys.argv)
    out = own.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = []
    printed = NativeResults()
    ns = {}
    error = None
    metadata = {'purpose': 'course_100_rollout_self_evaluation_with_telemetry',
                'training_seed': own.training_seed, 'status': 'running',
                'started_at': utc_now(), 'command': original_command}
    hashes_before = {}
    write_json(out / 'run.json', metadata)
    try:
        checked = preflight(native_args)
        if not checked.video_folder.is_relative_to(out):
            raise ValueError('--video_folder must be inside this new --output-dir')
        native = own.native_play.resolve(strict=True)
        hashes_before = {str(p): sha(p) for p in [*source_paths(native), checked.checkpoint]}
        metadata.update({
            'task': checked.task, 'evaluation_seed': checked.seed, 'num_rollouts': checked.num_rollouts,
            'horizon': checked.horizon, 'video': checked.video, 'video_length': checked.video_length,
            'video_folder': str(checked.video_folder),
            'checkpoint': str(checked.checkpoint), 'checkpoint_sha256': hashes_before[str(checked.checkpoint)],
            'native_play': str(native), 'native_play_sha256': hashes_before[str(native)],
            'observer_sha256': hashes_before[str(Path(__file__).resolve())],
            'source_and_checkpoint_sha256_before': hashes_before,
            'native_command': [str(native), *native_args],
            'note': 'Course self-validation with fixed evaluation seed 42, 100 rollouts and horizon 800; not the hidden-seed final evaluation. Video is enabled. Per-model trial numbers are not assumed paired.',
            'telemetry_columns': {
                'toy_root_state_w': [0,13], 'box_root_state_w': [13,26],
                'ee_position_w': [26,29], 'ee_quaternion_wxyz': [29,33],
                'gripper_joints': [33,35], 'pad_net_forces_w': [35,41],
                'robot_position_w': [41,44], 'robot_quaternion_wxyz': [44,48],
            },
            'state_alignment': 'state[0] is reset initial state; state[k+1] is after action[k]. For termination auto-reset, stored states contain captured pre-reset state; the returned observation is still post-reset. Native execution receives unchanged returns.',
        })
        write_json(out / 'run.json', metadata)
        sys.argv = [str(native), *native_args]
        ns = {'__name__': 'telemetry_native', '__file__': str(native), 'print': printed.print}
        exec(compile(native.read_bytes(), str(native), 'exec'), ns)
        import numpy as np
        import torch
        validate_settings(ns['args_cli'])
        native_rollout = ns['rollout']
        ns['rollout'] = observed_rollout
        ns['main']()
        printed.validate(records)
        disk_records = [json.loads(line) for line in (out / 'trials.jsonl').read_text().splitlines()]
        if disk_records != records:
            raise ValueError('Persisted trials.jsonl differs from in-memory records')
        for record in records:
            if sha(out / record['trajectory']) != record['trajectory_sha256']:
                raise ValueError(f"Changed trajectory file: {record['trajectory']}")
        metadata['native_results_and_telemetry_validated'] = True
        metadata['status'] = 'completed'
    except BaseException as exc:
        error = exc
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        hashes_after, hash_errors = {}, {}
        for path in hashes_before:
            try:
                hashes_after[path] = sha(path)
            except Exception as exc:
                hash_errors[path] = f'{type(exc).__name__}: {exc}'
        unchanged = bool(hashes_before) and hashes_before == hashes_after and not hash_errors
        if hashes_before and not unchanged:
            metadata.update(status='failed', integrity_error='Source or checkpoint changed during evaluation')
        metadata.update({
            'finished_at': utc_now(), 'recorded_trials': len(records),
            'successful_trials': sum(r['success'] for r in records),
            'complete_trial_indices': [r['trial'] for r in records],
            'source_and_checkpoint_sha256_after': hashes_after,
            'source_and_checkpoint_unchanged': unchanged,
            'source_hash_errors': hash_errors,
            'native_count_summaries': printed.counts, 'native_success_rates': printed.rates,
        })
        if metadata['status'] == 'completed':
            metadata['success_rate'] = metadata['successful_trials'] / ROLLOUTS
        else:
            metadata['partial_success_rate'] = metadata['successful_trials'] / len(records) if records else None
        write_json(out / 'summary.json', metadata)
        try:
            if ns.get('simulation_app') is not None:
                ns['simulation_app'].close()
        except BaseException as exc:
            metadata.update(status='failed', close_error=f'{type(exc).__name__}: {exc}')
            write_json(out / 'summary.json', metadata)
        sys.argv = original_command
    if metadata['status'] != 'completed':
        print(f"Evaluation failed; preserved records: {out}; {metadata.get('error', metadata.get('integrity_error', metadata.get('close_error')))}", file=sys.stderr)
        return 1
    print(f"Validated {metadata['successful_trials']}/{ROLLOUTS}; records: {out}", flush=True)
    return 0


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
