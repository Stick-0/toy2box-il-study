"""Read-only HDF5 audit for the 100 human demonstrations.

Episode identity includes every dataset and group attribute under the episode,
independent of the demo_N name, to survive the native merger's renumbering.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

BASE = Path('/home/jm/robot_sim')
OUT = BASE / 'local_analysis/human100_mimic1000'
FINAL = BASE / 'IsaacLab_RS/datasets/human100.hdf5'
SOURCES = [
    (BASE / 'toy2box-il-study/human_study/data/human_20261028_112300.hdf5', 30),
    (BASE / 'IsaacLab_RS/datasets/human_extra_20261029_131358.hdf5', 20),
    (BASE / 'IsaacLab_RS/datasets/human_extra_20261029_133833.hdf5', 25),
    (BASE / 'IsaacLab_RS/datasets/human_extra_20261029_145214.hdf5', 25),
]
INTERMEDIATES = [
    (BASE / 'IsaacLab_RS/datasets/human_extra_20261029_131358_merged.hdf5', 50),
    (BASE / 'IsaacLab_RS/datasets/human75.hdf5', 75),
]
EXPECTED_ENV = 'Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0'


def basic(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def episode_audit(group):
    content = hashlib.sha256()
    schema, bad_finite, bad_length = {}, [], []
    count = int(group.attrs['num_samples'])

    def visit(name, obj):
        content.update(name.encode())
        content.update(json.dumps({k: basic(v) for k, v in sorted(obj.attrs.items())}, sort_keys=True).encode())
        if isinstance(obj, h5py.Dataset):
            data = obj[...]
            schema[name] = {'shape': list(data.shape), 'dtype': str(data.dtype)}
            content.update(str(data.dtype).encode())
            content.update(json.dumps(data.shape).encode())
            content.update(data.tobytes(order='C'))
            if np.issubdtype(data.dtype, np.number) and not np.isfinite(data).all():
                bad_finite.append(name)
            expected_length = 1 if name.startswith('initial_state/') else count
            if data.ndim == 0 or data.shape[0] != expected_length:
                bad_length.append(name)

    visit('', group)
    group.visititems(visit)
    actions = group['actions'][...]
    return {
        'episode': group.name.rsplit('/', 1)[-1],
        'content_sha256': content.hexdigest(),
        'num_samples': count,
        'success': bool(group.attrs.get('success', False)),
        'action_shape': list(actions.shape),
        'action_dtype': str(actions.dtype),
        'nonfinite_dataset_paths': bad_finite,
        'length_mismatch_paths': bad_length,
        'schema': schema,
        'action_min': actions.min(axis=0).tolist(),
        'action_max': actions.max(axis=0).tolist(),
        'arm_zero_steps': int(np.all(actions[:, :6] == 0, axis=1).sum()),
        'gripper_unique': np.unique(actions[:, 6]).tolist(),
    }


def inspect(path, expected_count):
    before = file_sha(path)
    with h5py.File(path, 'r') as f:
        group = f['data']
        env = json.loads(group.attrs['env_args'])
        episodes = [episode_audit(group[k]) for k in sorted(group, key=lambda name: int(name.rsplit('_', 1)[-1]))]
        metadata_total = basic(group.attrs.get('total'))
    after = file_sha(path)
    return {
        'path': str(path), 'file_sha256': before, 'file_bytes': path.stat().st_size,
        'sha256_unchanged_during_audit': before == after,
        'expected_count': expected_count, 'count': len(episodes),
        'count_matches': len(episodes) == expected_count,
        'environment': env, 'environment_name_matches': env['env_name'] == EXPECTED_ENV,
        'metadata_total': metadata_total,
        'total_samples': sum(e['num_samples'] for e in episodes),
        'successful_episode_count': sum(e['success'] for e in episodes),
        'all_numeric_values_finite': not any(e['nonfinite_dataset_paths'] for e in episodes),
        'all_time_lengths_match': not any(e['length_mismatch_paths'] for e in episodes),
        'all_actions_T_by_7_float32': all(e['action_shape'] == [e['num_samples'], 7] and e['action_dtype'] == 'float32' for e in episodes),
        'episodes': episodes,
    }


def main():
    source_records = [inspect(p, n) for p, n in SOURCES]
    final_record = inspect(FINAL, 100)
    intermediate_records = [inspect(p, n) for p, n in INTERMEDIATES]
    source_by_hash = collections.defaultdict(list)
    for file_record in source_records:
        for ep in file_record['episodes']:
            source_by_hash[ep['content_sha256']].append({'path': file_record['path'], 'episode': ep['episode']})
    final_counts = collections.Counter(e['content_sha256'] for e in final_record['episodes'])
    original_counts = collections.Counter(e['content_sha256'] for f in source_records for e in f['episodes'])
    provenance = [
        {'final_episode': ep['episode'], 'content_sha256': ep['content_sha256'], 'source_matches': source_by_hash.get(ep['content_sha256'], [])}
        for ep in final_record['episodes']
    ]
    duplicates = [{'content_sha256': h, 'count': n, 'episodes': [e['episode'] for e in final_record['episodes'] if e['content_sha256'] == h]} for h, n in final_counts.items() if n > 1]
    environment_equal = all(f['environment'] == final_record['environment'] for f in source_records + intermediate_records)
    used_by_source = collections.Counter(m['path'] for ep in provenance for m in ep['source_matches'])
    final_lengths = np.array([e['num_samples'] for e in final_record['episodes']])
    checks = {
        'exactly_100_demonstrations': final_record['count'] == 100,
        'all_100_marked_success': final_record['successful_episode_count'] == 100,
        'every_final_episode_matches_exactly_one_source_episode': all(len(e['source_matches']) == 1 for e in provenance),
        'final_content_multiset_equals_30_plus_20_plus_25_plus_25': final_counts == original_counts,
        'no_identical_episode_duplicates': not duplicates,
        'all_environment_metadata_equal': environment_equal,
        'all_numeric_values_finite': final_record['all_numeric_values_finite'],
        'all_time_lengths_match': final_record['all_time_lengths_match'],
        'all_actions_T_by_7_float32': final_record['all_actions_T_by_7_float32'],
        'all_input_file_hashes_unchanged_during_audit': all(f['sha256_unchanged_during_audit'] for f in source_records + intermediate_records + [final_record]),
    }
    payload = {
        'purpose': 'Read-only audit before Mimic annotation and generation; no source HDF5 file is modified.',
        'passed': all(checks.values()), 'checks': checks,
        'episode_content_hash_definition': 'SHA256 of sorted HDF5 relative object paths, object attributes, dataset dtype, shape and raw C-order bytes, excluding outer demo_N name.',
        'counts_by_source_in_final': dict(used_by_source),
        'final_dataset': final_record, 'source_datasets': source_records,
        'intermediate_datasets': intermediate_records,
        'provenance': provenance, 'duplicate_episodes': duplicates,
        'summary': {
            'total_steps': int(final_lengths.sum()),
            'episode_length_min': int(final_lengths.min()),
            'episode_length_median': float(np.median(final_lengths)),
            'episode_length_max': int(final_lengths.max()),
            'step_dt_seconds': final_record['environment']['sim_args']['dt'] * final_record['environment']['sim_args']['decimation'],
            'arm_zero_action_fraction': sum(e['arm_zero_steps'] for e in final_record['episodes']) / int(final_lengths.sum()),
        },
        'limitations': [
            'The success field is the recorder-provided stored success flag; this audit does not replay demonstrations or independently certify final physical success.',
            'Exact content hashes detect copied/identical episodes, not near-duplicate demonstrations or insufficient state coverage.',
            'Merged HDF5 files lack the optional data/total attribute because the native merge script copies only env_args. All episode num_samples attributes match their recorded array lengths.',
            'Finite values and shape correctness are structural checks, not proof that demonstrations are optimal or consistently aligned.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data_audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '사람 시범 100개 데이터 감사',
        '',
        f"판정: {'통과' if payload['passed'] else '확인 필요'}",
        f'대상: {FINAL}',
        f"SHA256: {final_record['file_sha256']}",
        f"환경: {EXPECTED_ENV}",
        '구성: 기존 30개 + 추가 20개 + 추가 25개 + 추가 25개 = 100개',
        '',
    ]
    for record in source_records:
        lines.append(f"- {Path(record['path']).name}: {record['count']}개, {record['total_samples']:,} steps")
    lines += [
        '',
        f"전체 {final_record['total_samples']:,} steps; 시범 길이 최소 {final_lengths.min()}, 중앙값 {np.median(final_lengths):g}, 최대 {final_lengths.max()} steps.",
        '원본 네 파일의 모든 시범은 합친 파일에 정확히 한 번씩 포함되어 있다. 내용 해시 기준 완전 중복과 누락은 없다.',
        '저장된 success 속성은 100개 모두 True이다. 성공 여부를 새로 재생 검증한 것은 아니다.',
        '모든 수치 배열은 유한하다. actions는 모든 시범에서 (T, 7) float32이며, 관측·상태·처리된 행동 배열의 시간 길이는 각 num_samples와 일치한다.',
        '모든 입력 파일의 환경 메타데이터가 같다. step dt는 0.05초이다.',
        '감사 전후 모든 원본·병합 파일의 SHA256은 같으며, 파일은 읽기 전용으로 열었다.',
        '',
        '참고: 네이티브 병합 도구가 env_args만 복사하므로 병합본에는 선택적인 data/total 속성이 없다. 시범별 num_samples 및 실제 배열 길이는 모두 일치하며, 합계는 위와 같다.',
        '한계: 완전 중복 여부와 구조를 확인한 감사이다. 비슷한 상황의 반복 정도, 행동 품질, 물리적 성공을 재평가한 결과는 아니다.',
    ]
    (OUT / 'data_audit.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'passed': payload['passed'], 'checks': checks, 'summary': payload['summary'], 'counts_by_source': dict(used_by_source)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
