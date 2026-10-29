#!/usr/bin/env python3
"""Summarize completed, recorded seed-101/111 evaluation without new rollouts."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SEEDS = (101, 111)
PRIOR = Path('/home/jm/robot_sim/toy2box-il-study/human_study/results/mimic500_20261028/evaluation')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    annotation=json.loads((ROOT/'annotation_manifest.json').read_text())
    generation=json.loads((ROOT/'generation_summary.json').read_text())
    prior_rates={};prior_sources=[]
    for condition in ('baseline','mimic500'):
        prior_rates[condition]=[]
        for seed in SEEDS:
            folder=PRIOR/f'{condition}_seed{seed}'
            record=json.loads((folder/'summary.json').read_text())
            assert record['status']=='completed' and record['evaluation_seed']==42
            assert record['num_rollouts']==100 and record['horizon']==800
            trials=[json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()]
            assert [r['trial'] for r in trials]==list(range(100))
            count=sum(r['success'] for r in trials)
            assert count==record['successful_trials']
            prior_rates[condition].append(float(count))
            prior_sources.append({'summary':str(folder/'summary.json'),'sha256':digest(folder/'summary.json'),
                                  'trials_sha256':digest(folder/'trials.jsonl'),'video':record['video']})
    previous_mean=statistics.mean(prior_rates['mimic500'])
    previous_std=statistics.stdev(prior_rates['mimic500'])
    dataset = json.loads((ROOT/'dataset_manifest.json').read_text())
    assert dataset['output']['human_count'] == 100
    assert dataset['output']['mimic_count'] == 1000
    assert dataset['output']['demo_count'] == 1100
    assert digest(dataset['output']['path']) == dataset['output']['sha256']
    results = []
    for seed in SEEDS:
        folder = ROOT/'evaluation'/f'seed{seed}'
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['status'] == 'completed'
        assert summary['evaluation_seed'] == 42
        assert summary['num_rollouts'] == 100 and summary['horizon'] == 800
        rows = [json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()]
        assert [r['trial'] for r in rows] == list(range(100))
        successes = sum(r['success'] for r in rows)
        assert successes == summary['successful_trials']
        assert summary['training_seed'] == seed
        assert digest(summary['checkpoint']) == summary['checkpoint_sha256']
        results.append({'training_seed': seed, 'successes': successes, 'rollouts': 100,
                        'success_rate_percent': float(successes),
                        'checkpoint': summary['checkpoint'],
                        'checkpoint_sha256': summary['checkpoint_sha256']})
    rates = [r['success_rate_percent'] for r in results]
    result = {
        'human_count':100, 'mimic_count':1000, 'training_count':1100,
        'annotated_source_count':annotation['exported_count'],
        'annotation_skipped_count':annotation['skipped_count'],
        'generation_native_counts':generation['last_native_counts'],
        'training_total_steps':dataset['output']['total_steps'],
        'training_seeds':list(SEEDS), 'evaluation_seed':42, 'rollouts_per_model':100,
        'horizon':800, 'mean_percent':statistics.mean(rates),
        'sample_std_percent':statistics.stdev(rates), 'std_ddof':1,
        'mean_difference_from_prior_augmented_pp':statistics.mean(rates)-previous_mean,
        'prior_result_sources':prior_sources,
        'models':results, 'dataset_manifest_sha256':digest(ROOT/'dataset_manifest.json'),
        'comparison_limitation':'Both human and synthetic demonstration counts changed versus human30+Mimic500; this is not a one-factor causal comparison.',
        'recording_difference':'Current evaluation records videos and telemetry; prior evaluation had video disabled. Native environment, rollout, success, termination and model execution logic are preserved.',
        'checkpoint_selection':'Fixed final epoch 2000 for both seeds; no evaluation-based model selection.',
    }
    (ROOT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    prior=[('Human 30',prior_rates['baseline']),('Human 30 + Mimic 500',prior_rates['mimic500']),('Human 100 + Mimic 1000',rates)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    means=[statistics.mean(v) for _,v in prior];stds=[statistics.stdev(v) for _,v in prior]
    ax.bar(range(3),means,yerr=stds,capsize=5,color=['#abb5be','#86b7d6','#147d92'],alpha=.9)
    for i,(_,values) in enumerate(prior):
        for offset,value,seed,color in zip((-.06,.06),values,SEEDS,('#152238','#b85c16')):
            ax.scatter(i+offset,value,color=color,zorder=4,label=f'Training seed {seed}' if i==0 else None)
        ax.text(i,max(values)+4,f'{means[i]:.1f} ± {stds[i]:.2f}%',ha='center',fontsize=11)
    ax.set_xticks(range(3),[n.replace(' + ','\n+ ') for n,_ in prior]);ax.set_ylim(0,105)
    ax.set_ylabel('Success rate (%)');ax.set_title('Evaluation seed 42 · 100 rollouts per model · horizon 800')
    ax.legend(loc='upper left',frameon=False,title='Mean ± sample SD (n=2)')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'success_rates.png',dpi=180);plt.close(fig)
    text=(
        '사람 시범 100개 + Mimic 성공 시범 1,000개 학습·평가 결과\n\n'
        f'사람 100개 중 {annotation["exported_count"]}개가 자동 주석을 통과해 증강 원본으로 사용 가능했다. '
        f'Mimic은 {generation["last_native_counts"][1]:,}번 시도에서 성공 1,000개를 생성했다. '
        f'학습 합본에는 사람 원본 100개를 모두 포함하여 1,100개·{dataset["output"]["total_steps"]:,} steps를 사용했다.\n\n'
        f'학습 seed 101: {rates[0]:.0f}/100 ({rates[0]:.0f}%)\n'
        f'학습 seed 111: {rates[1]:.0f}/100 ({rates[1]:.0f}%)\n'
        f'평균 ± 표본 표준편차: {result["mean_percent"]:.1f} ± {result["sample_std_percent"]:.2f}% (n=2, ddof=1)\n\n'
        '평가 seed 42, 모델당 100회, horizon 800. 환경·성공 판정·종료 조건은 기존 강의 코드를 유지했다. '
        '평가 영상, 매 시행의 상태·행동과 성공 라벨, 원본 출력 로그를 함께 보관했다.\n'
        '학습은 기존과 같은 2,000 epoch, epoch당 100 update, batch 100이며 마지막 epoch 2000 모델을 사용했다.\n\n'
        f'기존 사람30+Mimic500의 {previous_mean:.1f} ± {previous_std:.2f}% 대비 평균 변화: {result["mean_difference_from_prior_augmented_pp"]:+.1f}%p.\n'
        '사람 시범 수와 증강 시범 수를 함께 변경했으므로 변화량을 어느 한 요인의 인과 효과로 해석하지 않는다. '
        '이전 결과는 당시 저장된 100회 평가 결과이며 이번에 이전 모델을 재평가하지 않았다. '
        '이번 평가는 영상·상태 기록을 활성화했고 이전 평가는 영상 기록을 비활성화했다. '
        '원본 환경·rollout·성공·종료·모델 실행 논리는 유지했다. 비공개 평가 seed의 성능은 확인하지 않았다.\n\n'
        '근거: dataset_manifest.json, training/seed*/summary.json, evaluation/seed*/summary.json, '
        'evaluation/seed*/trials.jsonl, evaluation/seed*/videos/, results.csv, success_rates.png\n'
        'GitHub 업로드는 수행하지 않았다.\n')
    (ROOT/'results.txt').write_text(text)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
