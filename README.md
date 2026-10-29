# Agibot Toy2Box: 물체 획득 실패 진단과 Mimic 증강

**연구 질문:** 사람 시범으로 학습한 BC 정책은 Toy2Box에서 왜 성공률이 저조한가? 원인 하나를 진단하고, 그 원인을 겨냥해 성공률을 높여라.

접근 → 정렬·그리퍼 회전 → 하강·파지 → 들어 올리기·운반 → 배치의 규칙으로 수집한 키보드 시범 30개를 사용했지만, 강의 설정의 Baseline 성공률은 **17.5 ± 2.12%**였다. 기존 녹화 영상과 HDF5를 분석해 **물체 획득 이전의 반복적인 실패**, **30개 시범으로는 드문드문 관측되는 초기 위치·방향 조합**을 확인했다. 이를 바탕으로 제한된 상태 조합에 대한 일반화 부족을 원인 가설로 검토한다.

같은 사람 시범에 Mimic 성공 시범 500개를 추가하고 학습·평가 조건을 유지한 결과, 성공률은 **62.0 ± 2.83%**로 증가했다. 두 학습 seed 모두 개선됐으며 평균 증가 폭은 **44.5%p**다. 증강의 개선 효과는 확인되었지만, 데이터 수·상태 분포·생성 행동의 변화가 함께 작용하므로 다양성만의 인과 효과는 분리되지 않았다.

사람 시범 **100개와 Mimic 성공 시범 1,000개**를 사용한 조건은 **66.5 ± 2.12%**를 기록했다. 30+500개 조건보다 seed 101은 4%p, seed 111은 5%p 높았다. 사람 수와 증강 수를 함께 변경한 **데이터 규모 비교**이므로, 이 4.5%p 차이를 사람 시범 추가 또는 Mimic 증량 하나의 효과로 해석하지 않는다. 최종 제출 모델은 100+1,000개 조건의 두 모델이다.

## 실험 결과

| 조건 | 학습 seed 101 | 학습 seed 111 | 평균 ± 표본 SD |
|---|---:|---:|---:|
| 사람 시범 30개 | 16/100 (16%) | 19/100 (19%) | 17.5 ± 2.12% |
| 동일 사람 30개 + Mimic 성공 500개 | 64/100 (64%) | 60/100 (60%) | 62.0 ± 2.83% |
| 사람 100개 + Mimic 성공 1,000개 | 68/100 (68%) | 65/100 (65%) | **66.5 ± 2.12%** |

평가 seed **42**, 모델당 **100 rollout**, **horizon 800**을 사용했다. 평균과 표본 표준편차는 학습 seed 101·111 두 개로 계산했다(`ddof=1`, `n=2`). 세 조건 모두 **2,000 epoch · epoch당 100 update · batch 100**이다. 환경·성공 판정·종료 조건은 강의 코드를 유지했다. 30개·30+500개 조건은 영상 기록 없이, 100+1,000개 조건은 영상·상태 기록을 활성화해 평가했다. 비공개 평가 seed 성능과 통계적 유의성은 확인하지 않았다.

![세 데이터 조건의 성공률 비교](human_study/results/human100_mimic1000/success_rates.png)

- [연구 내용: Baseline → 실패 진단 → 가설 → 통제 실험 → 판정](human_study/results.md)
- [연구 설계와 판정 기준](human_study/protocol.md)
- [실행 환경·데이터·재현 절차](human_study/reproduce.md)
- [30개·30+500개 집계](human_study/results/mimic500_20261028/comparison.json) · [100+1,000개 결과 CSV](human_study/results/human100_mimic1000/results.csv) · [집계 JSON](human_study/results/human100_mimic1000/results.json)

## 원인 해석을 뒷받침하는 데이터

사람 시범 30개와 Mimic 성공 시범 500개의 행동·초기 상태 분포를 비교하고, 녹화 영상과 학습 로그를 함께 검토했다.

| 확인 항목 | 측정 결과 | 해석 |
|---|---|---|
| 사람 시범의 그리퍼 명령 순서 | 30/30개에서 한 번 닫고 한 번 열기 | 동작 순서의 일부 일관성을 확인했다. 정밀한 정렬 품질까지 보증하지는 않는다. |
| 사람 시범의 초기 위치·방향 조합 | 고정된 96개 구간 중 27개를 포함 | 위치와 방향을 함께 고려하면 관측되지 않은 조합이 많다. |
| Mimic 500개 추가 후 같은 구간 | 91/96개 | 초기 상태의 표본 밀도가 증가했다. |
| 표본 수를 30개로 맞춘 Mimic | 1,000회 부분집합의 평균 25.6개 구간 | 전체 점유 구간 수에서는 사람 30개가 불리하지 않다. 표본 수의 영향을 고려해야 한다. |

사람 자료의 초기 장난감 x 범위는 **0.503–0.797m**로, 환경 범위 0.45–0.80m의 낮은 x 쪽 약 **5.3cm**에는 초기 표본이 없었다. 증강 자료는 **0.450–0.799m**를 포함한다. 동수 비교에서 전체 구간 수가 비슷하다는 점과 이러한 국소 공백은 함께 고려해야 한다.

![초기 위치·방향 분포와 표본 수를 맞춘 비교](human_study/analysis/initial_pose_coverage.png)

구간은 강의 환경의 초기 장난감 범위를 x 4개 × y 3개 × yaw 8개로 나눈 것이다. 위 수치는 **초기 상태의 기하학적 분포**이며 정책 성공률이나 평가 중 방문한 상태의 범위가 아니다. 그림의 동수 부분집합 비교도 추가 학습 실험이 아니다. [분석 방법·민감도·수치](human_study/analysis/data_diagnosis.md), [학습·영상 출처 검증](human_study/analysis/evidence_audit.md)

## 녹화 영상으로 확인한 실패

기존 Baseline 영상에서 두 seed에 걸쳐 고정 간격으로 선택한 30개를 검토했다. 접근 실패 1개, 파지 실패 19개, 접근·파지 경계 불명 7개로 **물체 획득 이전의 실패가 27개**였다. 나머지는 시각적 완료 2개, 파지·운반 뒤 배치 결과를 확정할 수 없는 1개였다.

![Baseline의 빈손 이탈·물체 밀어냄·시각적 완료 장면](human_study/video_analysis/baseline_visual_evidence.jpg)

세 행 모두 **Baseline**의 장면이다. Baseline 성공률은 영상 촬영과 별도 실행에서 집계했으므로, 이 영상 표본의 완료 수를 성공률로 사용하지 않았다. 30+500개 조건에는 대응하는 평가 영상이 없다. [표본 선정·단계별 판정](human_study/video_analysis/README.md), [파지 실패 GIF](human_study/video_analysis/clips/contact_displacement.gif), [결과가 불확실한 배치 GIF](human_study/video_analysis/clips/placement_ambiguity.gif)

100+1,000개 조건에서는 **성공률을 집계한 동일 실행의 영상 200개와 상태·행동 기록 200개**를 보관했다. 아래는 seed 111 / trial 0의 마지막 프레임으로, 공식 성공 라벨과 박스 안에 놓인 장난감을 함께 확인할 수 있다. 성공·실패 각 첫 시행은 기록 확인용 예시이며, 전체 실패 경향을 대표하는 표본으로 사용하지 않는다.

![100+1,000개 모델의 성공 시행 마지막 프레임](human_study/results/human100_mimic1000/figures/seed111_trial000_success_final.png)

[실패 시행 마지막 프레임](human_study/results/human100_mimic1000/figures/seed101_trial000_failure_final.png) · [200회 판정 검증](human_study/results/human100_mimic1000/evaluation_audit.txt) · [데이터·평가 영상 Release](https://github.com/Stick-0/toy2box-il-study/releases/tag/human100-mimic1000)

## 제출 파일

| 항목 | 경로 |
|---|---|
| IsaacLab 프로젝트 | [IsaacLab/](IsaacLab/) |
| Toy2Box 환경 | [환경 설정](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/place_toy2box_rmp_rel_env_cfg.py) |
| 강의 학습·평가 코드 | [train.py](IsaacLab/scripts/imitation_learning/robomimic/train.py) · [play.py](IsaacLab/scripts/imitation_learning/robomimic/play.py) |
| 최종 모델: 학습 seed 101 | [models/human100_mimic1000_seed101.pth](models/human100_mimic1000_seed101.pth) |
| 최종 모델: 학습 seed 111 | [models/human100_mimic1000_seed111.pth](models/human100_mimic1000_seed111.pth) |
| 모델 해시·학습 기록 | [models/manifest.json](models/manifest.json) |
| LMS 제출용 평가 명령어 | [submission/evaluation_commands.txt](submission/evaluation_commands.txt) |
| 사람 시범·증강 데이터와 검증 해시 | [human_study/data/README.md](human_study/data/README.md) |

가중치는 마지막 epoch 2000의 최종 모델 **2개만** 포함한다. 평가 결과를 보고 중간 epoch를 선택하지 않았다. 두 모델은 PyTorch 체크포인트이며 각각 약 8MB다.

## 평가 명령어

강의 Isaac Sim 환경에서 제출된 `IsaacLab/source/` 패키지를 설치한 뒤 저장소 루트에서 다음 명령을 실행한다. 자세한 환경 설정은 [재현 절차](human_study/reproduce.md)에 있다.

```bash
conda activate lerobot-arena
export TERM=xterm
cd IsaacLab

./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint ../models/human100_mimic1000_seed101.pth \
  --seed 42 --num_rollouts 100 --horizon 800 --headless \
  --video --video_length 1000 \
  --video_folder ../videos/human100_mimic1000_seed101

./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint ../models/human100_mimic1000_seed111.pth \
  --seed 42 --num_rollouts 100 --horizon 800 --headless \
  --video --video_length 1000 \
  --video_folder ../videos/human100_mimic1000_seed111
```

`--seed`는 평가 seed다. 학습 seed는 각 체크포인트와 학습 JSON의 `train.seed`에 저장되어 있다. 비공개 최종 평가는 조교가 지정하는 seed로 수행하며, 이 저장소의 수치는 자체 검증 seed 42에 한정된다.

위 명령은 기록된 68%·65% 평가와 같은 영상 옵션을 사용한다. 상태·행동 기록을 함께 저장하는 절차는 [재현 안내](human_study/reproduce.md)에 있다.

## 과제 일정

- **2026년 11월 3일(화) 23:59, KST:** LMS 조별 제출 — GitHub 링크와 평가 명령어 파일.
- **2026년 11월 5일(목) 수업 시간:** 조별 5분 발표 및 비공개 평가 seed 결과 공개.

공지와 제출 조건은 [과제 안내](human_study/assignment_notice.md)에 정리했다. IsaacLab 및 Mimic의 원 라이선스와 저작권 표시는 프로젝트 폴더에 보존되어 있다.
