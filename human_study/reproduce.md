# 실행 환경과 재현 절차

모든 경로는 저장소를 복제한 위치를 기준으로 한다. 최종 평가에는 `models/human100_mimic1000_seed101.pth`와 `models/human100_mimic1000_seed111.pth`를 사용한다. 기존 사람 30개+Mimic 500개 학습 데이터는 [데이터 안내](data/README.md)의 Release에서 준비한다. 사람 100개+Mimic 1,000개 원본은 아직 외부 미배포 상태이므로 해당 학습 재현 명령은 원본 파일 확보 후 실행한다. 제출 모델 평가와 저장된 결과 검증에는 학습 원본이 필요하지 않다.

## 1. 실행 환경

실험 환경은 Python 3.11.16, Isaac Sim 5.1.0, PyTorch 2.7.0+cu128, robomimic 0.4.0이다. 버전 기록은 [software.json](results/mimic500_20261028/software.json)에 있다. NVIDIA GPU와 Isaac 5.1 클라우드 자산에 대한 네트워크 접근이 필요하다. Agibot·장난감·박스·테이블·RMPFlow 자산은 강의 환경이 지정한 경로에서 로드한다.

강의의 Isaac Sim 환경을 활성화한 뒤, **이 저장소의 `IsaacLab/source/`**를 editable 설치한다. 이미 다른 IsaacLab 복사본을 사용했다면 패키지 경로를 이 제출본으로 다시 지정해야 한다. Isaac Sim 자체 설치 절차는 [프로젝트 설치 안내](../IsaacLab/README.md)를 따른다.

```bash
conda activate lerobot-arena
export TERM=xterm
cd IsaacLab
./isaaclab.sh -i none
./isaaclab.sh -p -m pip install robomimic==0.4.0
cd ..

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

Conda 환경 이름은 설치 환경에 맞게 지정한다. 실험 PC에서는 시스템 C++ 런타임 충돌을 피하기 위해 `LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"`를 사용했다. 같은 충돌이 발생하는 환경에서 해당 파일 존재를 확인한 뒤 `export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"`를 적용한다.

## 2. 최종 모델 평가

[submission/evaluation_commands.txt](../submission/evaluation_commands.txt)에 강의 `play.py` 형식의 제출 명령이 있다. `cd IsaacLab` 기준 체크포인트는 `../models/human100_mimic1000_seed101.pth`, `../models/human100_mimic1000_seed111.pth`이다. 두 모델의 SHA256과 학습 조건은 [모델 manifest](../models/manifest.json)에 있다.

평가 출력을 파일과 회차별 JSON으로 기록하려면 저장소 루트에서 다음 명령을 사용한다. 기존 결과를 덮어쓰지 않도록 새 출력 폴더를 지정한다.

```bash
for TRAIN_SEED in 101 111; do
  python human_study/scripts/evaluate_recorded.py \
    --checkpoint "models/human100_mimic1000_seed${TRAIN_SEED}.pth" \
    --label "human100_mimic1000_seed${TRAIN_SEED}" \
    --output-dir "human_study/reproduced/evaluation/seed${TRAIN_SEED}" \
    --video-folder "human_study/reproduced/evaluation/seed${TRAIN_SEED}/videos" || break
done
```

이 도구는 강의 `play.py`를 **평가 seed 42, 모델당 100 rollout, horizon 800, headless**로 실행한다. 위 명령은 영상을 함께 저장한다. 화면 확인은 제출된 native 명령에서 `--headless`를 생략한다. 비공개 최종 평가는 조교가 지정한 seed를 사용한다.

### 상태·행동 기록을 포함한 평가

사람 100개+Mimic 1,000개 실험은 성공 여부와 영상 외에 매 스텝의 상태·행동도 기록했다. 같은 기록을 생성하려면 위 평가 대신 아래 명령을 사용한다. 두 방법 모두 native 평가 함수를 사용하며, 아래 래퍼는 반환값과 reset 직전 상태를 관찰해 저장한다.

```bash
for TRAIN_SEED in 101 111; do
  python human_study/results/human100_mimic1000/scripts/evaluate_with_telemetry.py \
    --output-dir "human_study/reproduced/evaluation_telemetry/seed${TRAIN_SEED}" \
    --training-seed "$TRAIN_SEED" \
    --native-play "$PWD/IsaacLab/scripts/imitation_learning/robomimic/play.py" \
    --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
    --checkpoint "$PWD/models/human100_mimic1000_seed${TRAIN_SEED}.pth" \
    --seed 42 --num_rollouts 100 --horizon 800 --headless \
    --video --video_length 1000 \
    --video_folder "$PWD/human_study/reproduced/evaluation_telemetry/seed${TRAIN_SEED}/videos" || break
done
```

`--video_folder`는 새 `--output-dir` 내부에 지정한다. 래퍼는 native 성공 판정·합계와 기록 파일을 대조하고 체크포인트·평가 코드 해시가 유지되는지 확인한다. **기존 사람 30개 및 사람 30개+Mimic 500개 평가는 영상 없이 실행했고, 사람 100개+Mimic 1,000개 평가는 영상과 상태·행동 기록을 활성화했다.** 이 실행 차이를 보존하며, 평가 조건과 성공 기준 자체는 같다.

## 3. 저장된 결과와 제출 파일 검증

```bash
python human_study/scripts/verify_submission.py
```

시뮬레이션을 실행하지 않고 세 조건·각 두 seed의 저장된 평가 결과, 최종 모델 두 개와 제출 평가 경로를 확인한다. 실험별 근거는 [사람 30개 및 사람 30개+Mimic 500개 기록](results/mimic500_20261028/)과 [사람 100개+Mimic 1,000개 기록](results/human100_mimic1000/)에 있다.

PyTorch가 설치된 환경에서는 `python human_study/scripts/verify_submission.py --check-weights`로 최종 가중치를 CPU에 불러와 내장 설정과 유한한 수치 여부도 확인할 수 있다. 전체 200회 평가 영상과 상태·행동 원본의 다운로드·복원 방법은 [데이터 안내](data/README.md)에 있다.

이전 `summarize_comparison.py --submission`은 사람 30개+Mimic 500개 모델을 제출하던 구성용 도구다. 현재 최종 모델의 검증에는 위 `verify_submission.py`를 사용한다. 기존 모델의 가중치와 중간 체크포인트는 제출 용량 조건에 따라 포함하지 않으며, 설정·학습 로그·평가 결과·해시는 보존한다.

실행 당시 JSON에 포함된 절대 경로는 원 실행의 파일 식별 정보다. 배포 모델은 `models/manifest.json`과 제출 평가 명령의 저장소 상대 경로를 사용한다. 결과 폴더의 `scripts/`는 원 실행 코드의 보존본이므로, 절대 경로가 들어 있는 파이프라인 스크립트를 다른 PC에서 그대로 실행하지 않는다. 이 문서의 저장소 상대 경로 명령으로 재현한다.

### 기존 데이터·영상 분석 재현

사람 30개 및 Mimic 500개 파일을 준비한 뒤 다음 명령으로 530개 시범의 행동·초기 상태 분포, 동수 부분집합 비교와 그림을 재생성할 수 있다. NumPy, SciPy, h5py, Matplotlib이 필요하다.

```bash
python human_study/analysis/analyze_demonstrations.py
```

이 명령은 HDF5를 읽기 전용으로 열며 시뮬레이션·학습·평가를 실행하지 않는다. [측정 정의와 한계](analysis/data_diagnosis.md), [영상 분석](video_analysis/README.md), [학습·영상 출처 검증](analysis/evidence_audit.md)에 분석 범위와 원본 영상의 보존 위치를 설명했다.

## 4. 데이터와 학습 재현

[데이터 안내](data/README.md)에 따라 사용할 원본과 Mimic 성공 데이터를 준비하고 SHA256을 확인한다. 결합 출력이 이미 존재하면 반복 생성하지 않는다. HDF5 메타데이터에 따라 합본의 파일 해시는 달라질 수 있으므로 원본 해시와 시범별 출처·개수도 함께 확인한다.

### 사람 100개+Mimic 1,000개

```bash
python human_study/scripts/prepare_mimic_training.py \
  --human human_study/data/human100.hdf5 \
  --mimic human_study/data/human100_mimic1000_raw.hdf5 \
  --output human_study/data/human100_plus_mimic1000.hdf5 \
  --manifest human_study/data/rebuilt_human100_mimic1000.json \
  --limit 1000

for TRAIN_SEED in 101 111; do
  python human_study/scripts/train_recorded.py \
    --seed "$TRAIN_SEED" \
    --dataset human_study/data/human100_plus_mimic1000.hdf5 \
    --label "reproduce_human100_mimic1000_seed${TRAIN_SEED}" \
    --baseline-config "human_study/results/mimic500_20261028/configs/baseline_seed${TRAIN_SEED}.json" \
    --output-dir "human_study/reproduced/training/human100_mimic1000_seed${TRAIN_SEED}" || break
done
```

사람 시범 100개 전체와 성공한 Mimic 시범 1,000개를 합쳐 **1,100개**를 학습한다. Annotation을 통과한 93개는 생성 소스의 수이며, 사람 학습 시범을 93개로 줄이지 않는다.

### 기존 사람 30개+Mimic 500개

```bash
python human_study/scripts/prepare_mimic_training.py \
  --human human_study/data/human_20261028_112300.hdf5 \
  --mimic human_study/data/human30_mimic500_raw_20261028_env8.hdf5 \
  --output human_study/data/human30_plus_mimic500_20261028.hdf5 \
  --manifest human_study/data/rebuilt_human30_mimic500.json \
  --limit 500

for TRAIN_SEED in 101 111; do
  python human_study/scripts/train_recorded.py \
    --seed "$TRAIN_SEED" \
    --dataset human_study/data/human30_plus_mimic500_20261028.hdf5 \
    --label "reproduce_mimic500_seed${TRAIN_SEED}" \
    --baseline-config "human_study/results/mimic500_20261028/configs/baseline_seed${TRAIN_SEED}.json" \
    --output-dir "human_study/reproduced/training/mimic500_seed${TRAIN_SEED}" || break
done
```

사람 30개만 사용하는 Baseline을 재현하려면 위 학습 명령의 `--dataset`을 `human_study/data/human_20261028_112300.hdf5`로 바꾸고, 실험 이름과 출력 폴더도 새로운 Baseline 이름으로 지정한다.

### 공통 학습 조건과 모델 선택

`train_recorded.py`는 강의 학습 JSON의 `train.seed`를 101·111로 지정하고, 종료 시 원래 파일을 복원한다. 공유 JSON을 잠그므로 **위처럼 두 seed를 순차 실행**한다. 원래 사람 100개+Mimic 1,000개 실행은 seed별 JSON을 분리한 래퍼로 두 학습을 동시에 실행했으며, 실제 적용된 학습 조건이 동일함을 [학습 감사](results/human100_mimic1000/training_audit.txt)로 확인했다.

세 조건 모두 **2,000 epoch, epoch당 100 update, batch 100**, 동일한 BC-RNN 구성과 관측·행동 표현을 사용한다. 평가 대상은 마지막 **`model_epoch_2000.pth`**로 고정하며, 평가 성공률로 중간 모델을 고르지 않는다. 새 실행의 `summary.json`에 기록된 `checkpoint` 경로를 평가 명령에 지정한다.

## 5. Mimic 생성 절차

강의 참고자료 6–8쪽과 8주차 43–44쪽의 annotation → generation 절차를 적용한다. Release의 성공 데이터를 사용하면 이 단계는 필요하지 않다. 이미 같은 이름의 파일이 있다면 새 출력 경로를 지정한다. 생성 결과의 난수·시뮬레이션 실행에 따라 시도 횟수와 실제 시범은 달라질 수 있다.

아래 명령은 저장소 루트에서 시작한다.

```bash
cd IsaacLab

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --auto --headless \
  --input_file ../human_study/data/human100.hdf5 \
  --output_file ../human_study/data/human100_annotated.hdf5

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --headless --num_envs 8 --generation_num_trials 1000 \
  --input_file ../human_study/data/human100_annotated.hdf5 \
  --output_file ../human_study/data/human100_mimic1000_raw.hdf5
```

`generation_guarantee=True`이므로 성공 1,000개를 채우는 설정이다. 저장된 실행에서는 annotation 93/100개가 통과했고, 2,680회 시도에서 성공 1,000개를 얻었다. 실패 시범은 학습에 포함하지 않는다.

기존 사람 30개+Mimic 500개를 다시 생성할 때는 같은 `IsaacLab` 폴더에서 다음을 실행한다.

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --auto --headless \
  --input_file ../human_study/data/human_20261028_112300.hdf5 \
  --output_file ../human_study/data/human30_annotated_20261028.hdf5

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --headless --num_envs 8 --generation_num_trials 500 \
  --input_file ../human_study/data/human30_annotated_20261028.hdf5 \
  --output_file ../human_study/data/human30_mimic500_raw_20261028_env8.hdf5
```

기존 저장된 실행의 annotation 통과 수는 28/30개, 생성 결과는 1,389회 중 성공 500개다. 실험의 결과와 해석은 [연구 결과](results.md)에, 시범별 출처는 각 실험 폴더의 `dataset_manifest.json`에 있다. 사람 시범 수와 Mimic 시범 수를 함께 늘린 비교이므로 성능 변화량을 한 요인만의 인과 효과로 해석하지 않는다.
