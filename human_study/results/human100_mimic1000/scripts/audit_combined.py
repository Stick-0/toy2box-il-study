#!/usr/bin/env python3
"""Independently audit the prepared 100-human + 1000-Mimic training file.

All HDF5 inputs are opened read-only. Run only after dataset_manifest.json has
been published by prepare_mimic_training.py. No simulator or GPU is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
TASK = 'Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0'
MIMIC_TASK = 'Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0'
OBS_DIMS = {
    'actions': 7, 'joint_pos': 34, 'joint_vel': 34,
    'eef_pos': 3, 'eef_quat': 4, 'gripper_pos': 2,
    'toy_truck_positions': 3, 'toy_truck_orientations': 4,
    'box_positions': 3, 'box_orientations': 4,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def basic(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def names(data):
    result = list(data)
    if not all(re.fullmatch(r'demo_\d+', name) for name in result):
        raise ValueError('Unexpected non-demo name in ' + data.file.filename)
    return sorted(result, key=lambda name: int(name[5:]))


def inspect_episode(group):
    full_hash, array_hash = hashlib.sha256(), hashlib.sha256()
    issues = []
    steps = int(group.attrs.get('num_samples', -1))
    value = np.asarray(group.attrs.get('success', None))
    valid_success = value.shape == () and value.dtype.kind in 'biuf' and value.item() in (0, 1)
    success = bool(value.item()) if valid_success else None
    if not valid_success:
        issues.append('invalid/missing success attribute')
    if steps <= 0:
        issues.append('invalid/missing num_samples')
    shapes = {}

    def visit(name, obj):
        full_hash.update(name.encode())
        full_hash.update(json.dumps({k: basic(v) for k, v in sorted(obj.attrs.items())}, sort_keys=True).encode())
        if not isinstance(obj, h5py.Dataset):
            return
        data = obj[...]
        shapes[name] = {'shape': list(data.shape), 'dtype': str(data.dtype)}
        if data.dtype.kind not in 'biufc':
            issues.append(name + ': nonnumeric dataset')
        elif not np.isfinite(data).all():
            issues.append(name + ': nonfinite value')
        expected_time = 1 if name.startswith('initial_state/') else steps
        if data.ndim == 0 or data.shape[0] != expected_time:
            issues.append(name + ': time length differs from num_samples')
        prefix = json.dumps([name, str(data.dtype), list(data.shape)]).encode()
        for digest in (full_hash, array_hash):
            digest.update(prefix)
            digest.update(data.tobytes(order='C'))

    visit('', group)
    group.visititems(visit)
    expected = {'actions': [steps, 7], **{'obs/' + k: [steps, n] for k, n in OBS_DIMS.items()}}
    for name, shape in expected.items():
        if shapes.get(name, {}).get('shape') != shape:
            issues.append(f'{name}: expected shape {shape}, got {shapes.get(name)}')
    if shapes.get('actions', {}).get('dtype') != 'float32':
        issues.append('actions: expected float32')
    return {
        'demo': group.name.rsplit('/', 1)[-1], 'num_samples': steps,
        'success': success, 'content_sha256': full_hash.hexdigest(),
        'arrays_sha256': array_hash.hexdigest(), 'issues': issues,
        'dataset_count': len(shapes),
    }


def inspect_file(path):
    with h5py.File(path, 'r') as file:
        data = file['data']
        episodes = {name: inspect_episode(data[name]) for name in names(data)}
        env = json.loads(data.attrs['env_args'])
        metadata_total = basic(data.attrs.get('total'))
    return {
        'path': str(path), 'count': len(episodes), 'environment': env,
        'metadata_total': metadata_total,
        'total_samples': sum(e['num_samples'] for e in episodes.values()),
        'successful_count': sum(e['success'] is True for e in episodes.values()),
        'episode_issue_count': sum(bool(e['issues']) for e in episodes.values()),
        'episodes': episodes,
    }


def audit(root):
    manifest_path = root / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    plan_path = root / 'run_plan.json'
    plan = json.loads(plan_path.read_text())
    paths = {kind: Path(manifest['sources'][kind]['path']).resolve() for kind in ('human', 'mimic')}
    paths['combined'] = Path(manifest['output']['path']).resolve()
    expected_paths = {
        'human': root / 'data/human100.hdf5',
        'mimic': root / 'data/human100_mimic1000_raw.hdf5',
        'combined': root / 'data/human100_plus_mimic1000.hdf5',
    }
    for kind, expected in expected_paths.items():
        if paths[kind] != expected.resolve():
            raise ValueError(f'{kind}: unexpected input path: {paths[kind]}')
    original_human = Path(plan['human_source']).resolve()
    generation_path = root / 'generation_summary.json'
    generation = json.loads(generation_path.read_text())
    protected = [*paths.values(), original_human, manifest_path, plan_path, generation_path]
    hashes_before = {str(p): sha(p) for p in protected}
    inspected = {kind: inspect_file(path) for kind, path in paths.items()}
    human, mimic, combined = (inspected[k] for k in ('human', 'mimic', 'combined'))
    human_names = list(human['episodes'])
    successful_mimic_names = [name for name, e in mimic['episodes'].items() if e['success'] is True]
    selected_mimic_names = successful_mimic_names[:1000]
    expected_pairs = [('human', n) for n in human_names] + [('mimic', n) for n in selected_mimic_names]
    expected_provenance = [
        {'output_demo': f'demo_{i}', 'source': kind, 'source_file': str(paths[kind]),
         'source_demo': name, 'num_samples': inspected[kind]['episodes'][name]['num_samples']}
        for i, (kind, name) in enumerate(expected_pairs)
    ]
    copies = []
    for i, (kind, name) in enumerate(expected_pairs):
        source = inspected[kind]['episodes'][name]
        target = combined['episodes'].get(f'demo_{i}')
        copies.append({
            'output_demo': f'demo_{i}', 'source': kind, 'source_demo': name,
            'source_content_sha256': source['content_sha256'],
            'output_content_sha256': None if target is None else target['content_sha256'],
            'exact_copy': target is not None and target['content_sha256'] == source['content_sha256'],
            'num_samples': source['num_samples'],
        })
    array_groups = defaultdict(list)
    for name, ep in combined['episodes'].items():
        array_groups[ep['arrays_sha256']].append(name)
    duplicates = [{'arrays_sha256': h, 'episodes': group} for h, group in array_groups.items() if len(group) > 1]
    env = {kind: row['environment'] for kind, row in inspected.items()}
    sim_keys = ('dt', 'decimation', 'render_interval')
    checks = {
        'human_count_100': human['count'] == 100,
        'selected_successful_mimic_count_1000': len(selected_mimic_names) == 1000,
        'combined_count_1100': combined['count'] == 1100,
        'combined_names_exactly_demo0_to_demo1099': list(combined['episodes']) == [f'demo_{i}' for i in range(1100)],
        'all_combined_marked_success': combined['successful_count'] == 1100,
        'every_combined_episode_is_exact_content_copy': len(copies) == 1100 and all(e['exact_copy'] for e in copies),
        'human_100_copies_exact': len(human_names) == 100 and all(e['exact_copy'] for e in copies[:100]),
        'mimic_first_1000_successful_copies_exact': len(selected_mimic_names) == 1000 and all(e['exact_copy'] for e in copies[100:]),
        'no_duplicate_full_arrays_in_combined': not duplicates,
        'manifest_episode_provenance_exact': manifest['episodes'] == expected_provenance,
        'manifest_declares_100_1000_1100': all(manifest['output'][k] == v for k, v in {'human_count': 100, 'mimic_count': 1000, 'demo_count': 1100}.items()),
        'manifest_limit_1000': manifest['limit'] == 1000,
        'manifest_source_counts_match': all(manifest['sources'][kind]['demo_count'] == inspected[kind]['count'] for kind in ('human', 'mimic')),
        'manifest_mimic_success_count_matches': manifest['sources']['mimic']['successful_count'] == mimic['successful_count'],
        'manifest_hashes_match_files': all(manifest['sources'][kind]['sha256'] == hashes_before[str(paths[kind])] for kind in ('human', 'mimic')) and manifest['output']['sha256'] == hashes_before[str(paths['combined'])],
        'human_snapshot_matches_original_and_plan': hashes_before[str(paths['human'])] == hashes_before[str(original_human)] == plan['human_sha256'],
        'generation_raw_hash_matches': generation['raw_sha256'] == hashes_before[str(paths['mimic'])],
        'generation_raw_counts_match': generation['successful_file_count'] == mimic['successful_count'] and generation['raw_file_episode_count'] == mimic['count'],
        'all_sources_and_combined_arrays_finite_and_shapes_valid': all(f['episode_issue_count'] == 0 for f in inspected.values()),
        'combined_total_matches_arrays_and_manifest': combined['metadata_total'] == combined['total_samples'] == manifest['output']['total_steps'],
        'combined_total_equals_selected_source_sum': combined['total_samples'] == sum(e['num_samples'] for e in expected_provenance),
        'mimic_total_matches_all_raw_episode_lengths': mimic['metadata_total'] == mimic['total_samples'],
        'human_optional_total_matches_if_present': human['metadata_total'] in (None, human['total_samples']),
        'combined_environment_equals_original_human_environment': env['combined'] == env['human'],
        'environment_names_expected': env['human']['env_name'] == TASK and env['combined']['env_name'] == TASK and env['mimic']['env_name'] == MIMIC_TASK and manifest['output']['env_name'] == TASK,
        'simulation_dt_decimation_render_interval_match': all(env[kind]['sim_args'][key] == env['human']['sim_args'][key] for kind in ('mimic', 'combined') for key in sim_keys),
        'parallel_environment_counts_expected': env['human']['sim_args']['num_envs'] == 1 and env['combined']['sim_args']['num_envs'] == 1 and env['mimic']['sim_args']['num_envs'] == 8,
    }
    hashes_after = {str(p): sha(p) for p in protected}
    checks['all_inputs_unchanged_during_read_only_audit'] = hashes_before == hashes_after
    file_summaries = {kind: {k: v for k, v in row.items() if k != 'episodes'} for kind, row in inspected.items()}
    episode_issues = {kind: [ep for ep in row['episodes'].values() if ep['issues']] for kind, row in inspected.items()}
    return {
        'status': 'completed', 'passed': all(checks.values()),
        'audited_at': datetime.now(timezone.utc).isoformat(),
        'checks': checks, 'failed_checks': [name for name, passed in checks.items() if not passed],
        'source_and_output_summaries': file_summaries,
        'file_sha256_before': hashes_before, 'file_sha256_after': hashes_after,
        'manifest_sha256': hashes_before[str(manifest_path)],
        'duplicate_full_arrays': duplicates, 'episode_issues': episode_issues,
        'copy_verification': copies,
        'content_hash_definition': 'SHA256 over sorted relative HDF5 paths, all attributes, dataset names/dtypes/shapes and complete C-order bytes; outer demo_N name excluded.',
        'duplicate_hash_definition': 'SHA256 over dataset names/dtypes/shapes and complete C-order array bytes; all attributes and outer demo_N names excluded.',
        'limitations': [
            'success=True is the recorder-provided stored flag; this audit does not rerun physical success checks.',
            'Exact-array uniqueness does not establish scenario diversity or exclude near-duplicates.',
            'Mimic environment name and num_envs=8 are expected in the generation source. The combined training file retains the original human Toy2Box environment metadata.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        result = audit(root)
    except Exception as exc:
        result = {'status': 'failed', 'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
    (root / 'combined_data_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if result['status'] == 'completed':
        rows = result['source_and_output_summaries']
        lines = [
            '사람 시범 100개 + Mimic 성공 시범 1000개 합본 독립 감사', '',
            f"판정: {'통과' if result['passed'] else '확인 필요'}",
            f"총 {rows['combined']['count']}개 / {rows['combined']['total_samples']:,} steps",
            f"원본 사람 {rows['human']['count']}개; 생성 성공 {rows['mimic']['successful_count']}개 중 숫자 순 첫 1000개 선택",
            '검증 내용: 원본 전체 배열·속성의 정확한 복사, 완전 중복 여부, 수치 유효성·shape·길이, 환경 메타데이터, manifest·원본 SHA256 일치.',
            f"실패한 검사: {', '.join(result['failed_checks']) or '없음'}",
            '저장된 성공 플래그를 감사했으며 물리적 성공을 재실행하지 않았다. 완전 중복 검사는 상황 다양성에 대한 평가가 아니다.',
        ]
    else:
        lines = ['합본 독립 감사 실패', result['error']]
    (root / 'combined_data_audit.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'failed_checks', 'error') if k in result}, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
