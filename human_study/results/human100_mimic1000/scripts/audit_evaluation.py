#!/usr/bin/env python3
"""Audit existing native evaluation recordings using CPU-only calculations.

No simulator imports, evaluations, model inference, environment changes or Git
operations are performed. Only audit JSON/text files are written under root.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TASK = 'Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_predicate(path, function):
    """Compile just a pure native predicate, without importing Isaac Sim."""
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    namespace = {'torch': torch, 'SceneEntityCfg': lambda name: NS(name=name)}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[function]


def native_predicates(states, success_function, grasp_function):
    s = torch.from_numpy(states)
    robot = NS(data=NS(joint_pos=s[:, 33:35]), find_joints=lambda names: ([0, 1], []))
    scene = {
        'robot': robot,
        'toy_truck': NS(data=NS(root_pos_w=s[:, :3])),
        'box': NS(data=NS(root_pos_w=s[:, 13:16])),
        'ee_frame': NS(data=NS(target_pos_w=s[:, None, 26:29])),
        'contact_grasp': NS(data=NS(net_forces_w=s[:, 35:41].reshape(-1, 2, 3))),
    }
    env = NS(scene=scene, cfg=NS(gripper_joint_names=['right_hand_joint1', 'right_Right_1_Joint'], gripper_open_val=.994, gripper_threshold=.2))
    success = success_function(env, robot_cfg=NS(name='robot'), object_a_cfg=NS(name='toy_truck'), object_b_cfg=NS(name='box'), xy_threshold=.10, height_diff=.06, height_threshold=.04)
    grasp = grasp_function(env, robot_cfg=NS(name='robot'), ee_frame_cfg=NS(name='ee_frame'), object_cfg=NS(name='toy_truck'), diff_threshold=.05, force_threshold=1.)
    delta = s[:, :3] - s[:, 13:16]
    height = torch.linalg.vector_norm(delta[:, 2:], dim=1)
    opening_error = torch.abs(torch.abs(s[:, 33:35]) - .994)
    independent_success = (torch.linalg.vector_norm(delta[:, :2], dim=1) < .10) & ((height - .06) < .04) & (opening_error < .2).all(dim=1)
    independent_grasp = (torch.linalg.vector_norm(s[:, :3]-s[:, 26:29], dim=1) < .05) & (torch.linalg.vector_norm(s[:, 35:41].reshape(-1, 2, 3), dim=2) > 1.).all(dim=1) & (opening_error > .2).all(dim=1)
    check(torch.equal(success, independent_success), 'Native and independent success predicate differ')
    check(torch.equal(grasp, independent_grasp), 'Native and independent grasp predicate differ')
    return success.numpy(), grasp.numpy(), int(((height-.06 < .04) != (height < .10)).sum())


def completed_records(path, partial):
    raw = path.read_bytes() if path.exists() else b''
    tail = bool(raw and not raw.endswith(b'\n'))
    if tail:
        check(partial, 'Unterminated final JSONL record')
        raw = raw.rsplit(b'\n', 1)[0]+b'\n' if b'\n' in raw else b''
    return [json.loads(line) for line in raw.splitlines() if line], sha_bytes(raw), tail


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def audit_seed(root, seed, partial, model):
    folder = root/'evaluation'/f'seed{seed}'
    run = json.loads((folder/'run.json').read_text())
    for key, value in {'task': TASK, 'training_seed': seed, 'evaluation_seed': 42, 'num_rollouts': 100, 'horizon': 800, 'video': True}.items():
        check(run[key] == value, f'{seed}: invalid {key}')
    check(run['video_length'] >= 801, f'{seed}: incomplete video duration setting')
    check(Path(run['checkpoint']) == Path(model['path']), f'{seed}: model path differs from manifest')
    check(sha(run['checkpoint']) == model['sha256'] == run['checkpoint_sha256'], f'{seed}: checkpoint hash mismatch')
    cfg = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)['config']
    cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
    check(cfg['train']['seed'] == seed, f'{seed}: embedded training seed mismatch')
    before = run['source_and_checkpoint_sha256_before']
    check(all(sha(p) == h for p, h in before.items()), f'{seed}: current source/checkpoint hash mismatch')
    check(sha(run['native_play']) == run['native_play_sha256'], f'{seed}: native play mismatch')
    check(sha(root/'evaluate_with_telemetry.py') == run['observer_sha256'], f'{seed}: observer mismatch')
    lab = Path(run['native_play']).parents[3]
    mdp = lab/'source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/mdp'
    success_function = load_predicate(mdp/'terminations.py', 'object_a_is_into_b')
    grasp_function = load_predicate(mdp/'observations.py', 'object_grasped')
    records, records_hash, tail_ignored = completed_records(folder/'trials.jsonl', partial)
    check([r['trial'] for r in records] == list(range(len(records))), f'{seed}: non-contiguous trial indices')
    check(len(records) <= 100 and (partial or len(records) == 100), f'{seed}: trial count is {len(records)}')
    report = {'seed': seed, 'recorded_trials': len(records), 'successful_trials': sum(r['success'] for r in records), 'complete_jsonl_snapshot_sha256': records_hash, 'incomplete_jsonl_tail_ignored': tail_ignored, 'checkpoint_sha256': run['checkpoint_sha256'], 'source_hash_count': len(before), 'states': 0, 'actions': 0, 'native_grasp_states': 0, 'height_float_comparison_differences': 0, 'auto_reset_trials': [], 'maximum_alignment_errors': {}}
    for record in records:
        trial = record['trial']; label = f'{seed}/{trial:03d}'
        check(record['training_seed'] == seed and record['step_dt'] == .05, f'{label}: metadata mismatch')
        check(record['trajectory'] == f'trial_{trial:03d}.npz', f'{label}: unexpected trajectory path')
        path = folder/record['trajectory']
        check(sha(path) == record['trajectory_sha256'], f'{label}: trajectory hash mismatch')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        s, a, flags = arrays['states'], arrays['actions'], arrays['termination_flags']; n = len(a)
        check(s.shape == (n+1, 48) and a.shape == (n, 7) and flags.shape == (n, 3), f'{label}: invalid array shapes')
        check(s.dtype == np.float32 and flags.dtype == bool, f'{label}: unexpected state/flag dtype')
        check(n == record['steps'] and 0 < n <= 800, f'{label}: invalid step count')
        check(all(np.isfinite(v).all() for v in arrays.values()), f'{label}: non-finite values')
        check(all(len(v) == n+1 for k, v in arrays.items() if k.startswith('obs_')), f'{label}: observation length mismatch')
        check(bool(flags[:, 2].any()) == record['auto_reset'], f'{label}: reset metadata mismatch')
        check(np.array_equal(flags[:, 2], flags[:, :2].any(axis=1)), f'{label}: reset/termination inconsistency')
        if flags[:, 2].any():
            check(flags[-1, 2] and flags[:, 2].sum() == 1, f'{label}: reset not at terminal step')
            check(s[-1, 2] < -.85, f'{label}: pre-reset dropping state missing')
            report['auto_reset_trials'].append(trial)
        else:
            check(record['success'] or n == 800, f'{label}: early failure without termination')
        mask = np.r_[True, ~flags[:, 2]]
        for target, offset in [('toy_truck_positions', 0), ('box_positions', 13), ('eef_pos', 26)]:
            q = s[:, 44:48]; v = s[:, offset:offset+3]-s[:, 41:44]; qi = -q[:, 1:]
            relative = v+2*(q[:, 0, None]*np.cross(qi, v)+np.cross(qi, np.cross(qi, v)))
            error = float(np.max(np.abs(relative[mask]-arrays['obs_'+target][mask])))
            check(error < 1e-6, f'{label}: state/observation position mismatch {target}: {error}')
            report['maximum_alignment_errors'][target] = max(error, report['maximum_alignment_errors'].get(target, 0.))
        action_error = float(np.max(np.abs(a[~flags[:, 2]]-arrays['obs_actions'][1:][~flags[:, 2]])))
        check(action_error == 0, f'{label}: recorded raw actions differ from next policy obs_actions')
        report['maximum_alignment_errors']['raw_action_vs_next_obs_actions'] = action_error
        success, grasp, float_differences = native_predicates(s, success_function, grasp_function)
        check(bool(success[1:].any()) == record['success'], f'{label}: native success mismatch')
        if record['success']:
            check(success[-1] and not success[1:-1].any(), f'{label}: success did not stop at first native success')
        report['states'] += n+1; report['actions'] += n
        report['native_grasp_states'] += int(grasp.sum())
        report['height_float_comparison_differences'] += float_differences
    log_path = root/'logs'/f'evaluation_seed{seed}.log'
    log = log_path.read_text(errors='replace')
    native_trials = [(int(i), value == 'True') for i, value in re.findall(r'^\[INFO\] Trial (\d+): (True|False)\s*$', log, re.M)]
    expected = [(r['trial'], r['success']) for r in records]
    if partial:
        shared = min(len(expected), len(native_trials))
        check(native_trials[:shared] == expected[:shared], f'{seed}: native stdout labels disagree')
    else:
        check(native_trials == expected, f'{seed}: native stdout labels disagree')
        summary = json.loads((folder/'summary.json').read_text())
        check(summary['status'] == 'completed', f'{seed}: evaluation did not complete')
        check(summary['recorded_trials'] == 100 and summary['complete_trial_indices'] == list(range(100)), f'{seed}: final summary indices/count invalid')
        count = report['successful_trials']
        check(summary['successful_trials'] == count and summary['success_rate'] == count/100, f'{seed}: final summary outcome invalid')
        check(summary['native_count_summaries'] == [[count, 100]] and summary['native_success_rates'] == [count/100], f'{seed}: captured native totals mismatch')
        check(summary['native_results_and_telemetry_validated'] and summary['source_and_checkpoint_unchanged'], f'{seed}: native wrapper validation failed')
        check(summary['source_and_checkpoint_sha256_after'] == before and not summary['source_hash_errors'], f'{seed}: final source hashes differ')
        counts = re.findall(r'^Successful trials: (\d+), out of (\d+) trials\s*$', log, re.M)
        check(counts == [(str(count), '100')], f'{seed}: native raw totals mismatch')
        rates = re.findall(r'^Success rate: ([0-9.eE+-]+)\s*$', log, re.M)
        check(len(rates) == 1 and float(rates[0]) == count/100, f'{seed}: native raw rate mismatch')
        full_labels = re.findall(r'^Trial Results: (\[.*\])\s*$', log, re.M)
        check(len(full_labels) == 1 and ast.literal_eval(full_labels[0]) == [r['success'] for r in records], f'{seed}: native full array mismatch')
        check(len(list(folder.glob('trial_*.npz'))) == 100, f'{seed}: unexpected NPZ file count')
    report['native_printed_trials_available'] = len(native_trials)
    report['source_checks_passed'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    torch.set_num_threads(1)
    reference = json.loads((root/'native_source_hashes.json').read_text())
    check(all(sha(p) == h for p, h in reference.items()), 'Source differs from pre-experiment source manifest')
    models = json.loads((root/'models/manifest.json').read_text())['models']
    runs = [audit_seed(root, seed, args.allow_partial, next(m for m in models if m['training_seed'] == seed)) for seed in [101, 111]]
    report = {'status': 'partial_pass' if args.allow_partial else 'pass', 'scope': 'Read-only CPU audit of existing evaluation recordings; no simulator, inference, training, Git or environment changes.', 'auditor_sha256': sha(__file__), 'original_source_hash_count': len(reference), 'runs': runs, 'limitations': ['Grasp predicate is recomputed from saved state and net contact forces, not independently logged object-specific contact labels.', 'Terminal stored state is pre-reset; returned final observations after native termination are post-reset and excluded from alignment checks.', 'Success labels use the exact native predicate without a settling requirement.', 'Trial indices across models are not assumed paired. Video rendering is enabled in this evaluation.']}
    stem = 'evaluation_audit_partial' if args.allow_partial else 'evaluation_audit'
    (root/(stem+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    lines = [f'Evaluation audit: {report["status"]}', report['scope'], f'Original source hashes verified: {len(reference)}', '']
    for run in runs:
        lines.extend([f'Seed {run["seed"]}: {run["successful_trials"]}/{run["recorded_trials"]} successes; {run["actions"]} actions; {run["states"]} states.', 'Native stdout, exact source predicate, record continuity, file hashes, shapes, finite values and state/action alignment verified.', f'Auto-reset trials with pre-reset dropping state verified: {run["auto_reset_trials"]}', f'Maximum alignment errors: {run["maximum_alignment_errors"]}', f'Recomputed native grasp-positive states: {run["native_grasp_states"]}', ''])
    lines.extend(report['limitations'])
    (root/(stem+'.txt')).write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
