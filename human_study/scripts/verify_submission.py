#!/usr/bin/env python3
"""Verify the portable submission and six saved evaluations; no simulation runs."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'human_study/results/mimic500_20261028'
NEW = ROOT / 'human_study/results/human100_mimic1000'
SEEDS = (101, 111)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def normalized(config):
    config = copy.deepcopy(config)
    for key in ('data', 'output_dir'):
        config['train'].pop(key, None)
    config['experiment'].pop('name', None)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-weights', action='store_true',
                        help='Also load the hash-verified submitted checkpoints on CPU (requires PyTorch).')
    args = parser.parse_args()
    rates = {}
    for condition in ('baseline', 'mimic500', 'human100_mimic1000'):
        rates[condition] = []
        for seed in SEEDS:
            folder = (NEW / 'evaluation' / f'seed{seed}' if condition == 'human100_mimic1000'
                      else OLD / 'evaluation' / f'{condition}_seed{seed}')
            summary = read(folder / 'summary.json')
            require(summary['status'] == 'completed', f'{folder}: unfinished')
            require((summary['evaluation_seed'], summary['num_rollouts'], summary['horizon']) == (42, 100, 800),
                    f'{folder}: wrong evaluation conditions')
            trials = [json.loads(line) for line in (folder / 'trials.jsonl').read_text().splitlines()]
            require([x['trial'] for x in trials] == list(range(100)), f'{folder}: missing/duplicate trials')
            require(all(type(x['success']) is bool for x in trials), f'{folder}: non-boolean result')
            count = sum(x['success'] for x in trials)
            require(count == summary['successful_trials'] and abs(count / 100 - summary['success_rate']) < 1e-12,
                    f'{folder}: count/rate mismatch')
            rates[condition].append(count)

    result = read(NEW / 'results.json')
    require(result['mean_percent'] == statistics.mean(rates['human100_mimic1000']), 'Mean mismatch')
    require(abs(result['sample_std_percent'] - statistics.stdev(rates['human100_mimic1000'])) < 1e-12,
            'Sample SD mismatch')
    require(result['mean_difference_from_prior_augmented_pp'] ==
            statistics.mean(rates['human100_mimic1000']) - statistics.mean(rates['mimic500']), 'Delta mismatch')
    dataset = read(NEW / 'dataset_manifest.json')
    require(tuple(dataset['output'][x] for x in ('human_count', 'mimic_count', 'demo_count', 'total_steps')) ==
            (100, 1000, 1100, 434605), 'Dataset count mismatch')
    models = read(ROOT / 'models/manifest.json')['models']
    submission = read(ROOT / 'submission/manifest.json')
    require(models == submission['final_models'], 'Model manifests differ')
    require(sorted(x['training_seed'] for x in models) == list(SEEDS), 'Expected two final seeds')
    require(sorted(x.name for x in (ROOT / 'models').glob('*.pth')) ==
            sorted(Path(x['path']).name for x in models), 'Unexpected final model files')
    for model in models:
        seed = model['training_seed']
        path = ROOT / model['path']
        digest = sha(path)
        require(digest == model['sha256'], f'{path}: model hash mismatch')
        training = read(ROOT / model['training_record'])
        evaluation = read(ROOT / model['evaluation_record'])
        require(training['status'] == 'completed' and training['seed'] == seed, 'Training record mismatch')
        require(training['checkpoint_sha256'] == evaluation['checkpoint_sha256'] == digest,
                'Trained, evaluated and submitted weights differ')
        require(training['dataset_sha256'] == dataset['output']['sha256'], 'Training dataset mismatch')
        config = read(NEW / 'training' / f'seed{seed}' / 'effective_config.json')
        require((config['train']['seed'], config['train']['num_epochs'], config['train']['batch_size'],
                 config['experiment']['epoch_every_n_steps']) == (seed, 2000, 100, 100), 'Training setting mismatch')
        for previous in (OLD / 'configs' / f'baseline_seed{seed}.json',
                         OLD / 'training' / f'mimic500_seed{seed}' / 'effective_config.json'):
            require(normalized(config) == normalized(read(previous)), f'{previous}: controlled setting changed')
        if args.check_weights:
            import torch
            saved = torch.load(path, map_location='cpu', weights_only=False)
            embedded = saved['config']
            embedded = json.loads(embedded) if isinstance(embedded, str) else embedded
            require(embedded == config, f'{path}: embedded config mismatch')
            pending, tensor_count = [saved['model']], 0
            while pending:
                value = pending.pop()
                if torch.is_tensor(value):
                    require(bool(torch.isfinite(value).all()), f'{path}: non-finite weight')
                    tensor_count += 1
                elif isinstance(value, dict):
                    pending.extend(value.values())
                elif isinstance(value, (tuple, list)):
                    pending.extend(value)
            require(tensor_count > 0, f'{path}: no weight tensors checked')

    for path, digest in submission['source_sha256'].items():
        require(sha(ROOT / 'IsaacLab' / path) == digest, f'Source differs: {path}')
    registration = submission['registered_training_config_comparison']
    require(read(ROOT / 'IsaacLab' / registration['path']) == read(ROOT / registration['recorded_source_copy']),
            'Registered JSON settings differ')
    for record in read(NEW / 'archive_manifest.json')['files']:
        require(sha(ROOT / record['repository_path']) == record['sha256'],
                f'Archived record differs: {record["repository_path"]}')
    commands = (ROOT / submission['evaluation_commands']).read_text()
    paths = re.findall(r'--checkpoint\s+(\S+)', commands)
    require(len(paths) == 2 and {(ROOT / 'IsaacLab' / p).resolve() for p in paths} ==
            {(ROOT / model['path']).resolve() for model in models}, 'Evaluation command paths differ')
    print(json.dumps({'status': 'passed', 'recorded_rollouts_verified': 600,
                      'successes_per_100_by_seed_101_111': rates, 'final_model_count': 2,
                      'embedded_configs_checked_on_cpu': args.check_weights,
                      'new_simulation_rollouts': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
