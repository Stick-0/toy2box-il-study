# Agibot Toy2Box: 시범 다양성 확대를 통한 BC 성공률 개선

**연구 질문:** 사람 시범으로 학습한 BC 정책은 Toy2Box에서 왜 성공률이 저조한가? 원인 하나를 진단하고, 그 원인을 겨냥해 성공률을 높여라.

접근 → 정렬·그리퍼 회전 → 하강·파지 → 들어 올리기·운반 → 배치의 동작 규칙을 적용해 키보드 시범 30개를 수집했다. 그러나 강의 설정의 Baseline 성공률은 **17.5 ± 2.12%**였다. 실패 영상 표본에서 물체를 안정적으로 확보하기 전의 문제가 반복되어, **물체와 손의 상대 위치 변화에 대한 시범 다양성 부족**을 우선 원인 가설로 설정했다.

같은 사람 시범에 Mimic 성공 시범 500개를 추가하고 학습·평가 조건을 유지한 결과, 성공률은 **62.0 ± 2.83%**로 증가했다. 두 학습 seed 모두 개선됐으며 평균 증가 폭은 **44.5%p**다. 이는 증강의 개선 예측을 지지하지만, 수집 품질의 모든 문제를 배제하거나 다양성 부족을 유일한 원인으로 확정하는 결과는 아니다.

## 실험 결과

| 조건 | 학습 seed 101 | 학습 seed 111 | 평균 ± 표본 SD |
|---|---:|---:|---:|
| 사람 시범 30개 | 16/100 (16%) | 19/100 (19%) | 17.5 ± 2.12% |
| 동일 사람 30개 + Mimic 성공 500개 | 64/100 (64%) | 60/100 (60%) | **62.0 ± 2.83%** |
| 변화량 | +48%p | +41%p | **+44.5%p** |

평가 seed **42**, 모델당 **100 rollout**, **horizon 800**을 사용했다. 평균과 표본 표준편차는 학습 seed 101·111 두 개로 계산했다(`ddof=1`, `n=2`). 두 조건 모두 **2,000 epoch · epoch당 100 update · batch 100**이다. 환경·성공 판정·종료 조건은 강의 코드를 유지했다.

![Baseline과 Mimic 증강의 성공률](human_study/results/mimic500_20261028/figures/success_rates.png)

- [연구 내용: Baseline → 실패 진단 → 가설 → 통제 실험 → 판정](human_study/results.md)
- [연구 설계와 판정 기준](human_study/protocol.md)
- [실행 환경·데이터·재현 절차](human_study/reproduce.md)
- [결과 CSV](human_study/results/mimic500_20261028/comparison.csv) · [검증 기록 JSON](human_study/results/mimic500_20261028/comparison.json)

## 제출 파일

| 항목 | 경로 |
|---|---|
| IsaacLab 프로젝트 | [IsaacLab/](IsaacLab/) |
| Toy2Box 환경 | [환경 설정](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/place_toy2box_rmp_rel_env_cfg.py) |
| 강의 학습·평가 코드 | [train.py](IsaacLab/scripts/imitation_learning/robomimic/train.py) · [play.py](IsaacLab/scripts/imitation_learning/robomimic/play.py) |
| 최종 모델: 학습 seed 101 | [models/mimic500_seed101.pth](models/mimic500_seed101.pth) |
| 최종 모델: 학습 seed 111 | [models/mimic500_seed111.pth](models/mimic500_seed111.pth) |
| 모델 해시·학습 기록 | [models/manifest.json](models/manifest.json) |
| LMS 제출용 평가 명령어 | [submission/evaluation_commands.txt](submission/evaluation_commands.txt) |
| 사람 시범 30개·증강 데이터 안내 | [human_study/data/README.md](human_study/data/README.md) |

가중치는 마지막 epoch 2000의 최종 모델 **2개만** 포함한다. 평가 결과를 보고 중간 epoch를 선택하지 않았다. 두 모델은 PyTorch 체크포인트이며 각각 약 8MB다.

## 평가 명령어

강의 Isaac Sim 환경에서 제출된 `IsaacLab/source/` 패키지를 설치한 뒤 저장소 루트에서 다음 명령을 실행한다. 자세한 환경 설정은 [재현 절차](human_study/reproduce.md)에 있다.

```bash
conda activate lerobot-arena
export TERM=xterm
cd IsaacLab

./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint ../models/mimic500_seed101.pth \
  --seed 42 --num_rollouts 100 --horizon 800 --headless

./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint ../models/mimic500_seed111.pth \
  --seed 42 --num_rollouts 100 --horizon 800 --headless
```

`--seed`는 평가 seed다. 학습 seed는 각 체크포인트와 학습 JSON의 `train.seed`에 저장되어 있다. 비공개 최종 평가는 조교가 지정하는 seed로 수행하며, 이 저장소의 수치는 자체 검증 seed 42에 한정된다.

## 과제 일정

- **2026년 11월 3일(화) 23:59, KST:** LMS 조별 제출 — GitHub 링크와 평가 명령어 파일.
- **2026년 11월 5일(목) 수업 시간:** 조별 5분 발표 및 비공개 평가 seed 결과 공개.

공지와 제출 조건은 [과제 안내](human_study/assignment_notice.md)에 정리했다. IsaacLab 및 Mimic의 원 라이선스와 저작권 표시는 프로젝트 폴더에 보존되어 있다.
