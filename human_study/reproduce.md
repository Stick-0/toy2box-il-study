# 실행 환경과 재현 절차

모든 경로는 저장소를 복제한 위치를 기준으로 한다. 모델 평가에는 저장소의 최종 가중치 2개만 필요하며, 학습 재현에는 사람 시범 30개와 Release의 Mimic 성공 데이터가 필요하다.

## 1. 실행 환경

실험 환경은 Python 3.11.16, Isaac Sim 5.1.0, PyTorch 2.7.0+cu128, robomimic 0.4.0이었다. 전체 버전은 [software.json](results/mimic500_20261028/software.json)에 있다. NVIDIA GPU와 Isaac 5.1 클라우드 자산에 대한 네트워크 접근이 필요하다. Agibot·장난감·박스·테이블·RMPFlow 자산은 강의 환경이 지정한 경로에서 로드한다.

강의의 Isaac Sim 환경을 활성화한 뒤, **이 저장소의 `IsaacLab/source/`**를 editable 설치한다. 이미 다른 IsaacLab 복사본을 사용했다면 패키지 경로를 이 제출본으로 다시 지정해야 한다. Isaac Sim 자체 설치 절차는 [프로젝트 설치 안내](../IsaacLab/README.md)를 따른다.

```bash
conda activate lerobot-arena
export TERM=xterm
cd IsaacLab
./isaaclab.sh -i none
./isaaclab.sh -p -m pip install robomimic==0.4.0
cd ..
```

Conda 환경 이름은 설치 환경에 맞게 지정한다. 실험 PC에서는 시스템 C++ 런타임 충돌을 피하기 위해 `LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"`를 사용했다. 같은 충돌이 발생하는 환경에서 해당 파일 존재를 확인한 뒤 적용한다. 생성·학습의 CPU 스레드 수는 `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`였다.

## 2. 최종 모델 평가

[submission/evaluation_commands.txt](../submission/evaluation_commands.txt)에 강의 `play.py` 형식의 두 명령이 있다. `cd IsaacLab` 기준 체크포인트는 `../models/mimic500_seed101.pth`, `../models/mimic500_seed111.pth`이다. 두 모델의 SHA256과 학습 기록은 [모델 manifest](../models/manifest.json)에 있다.

평가 출력을 파일과 회차별 JSON으로 함께 기록하려면 저장소 루트에서 다음 도구를 사용한다. 기존 결과를 덮어쓰지 않도록 새 출력 폴더를 지정한다.

```bash
python human_study/scripts/evaluate_recorded.py \
  --checkpoint models/mimic500_seed101.pth \
  --label mimic500_seed101 \
  --output-dir human_study/reproduced/evaluation/mimic500_seed101

python human_study/scripts/evaluate_recorded.py \
  --checkpoint models/mimic500_seed111.pth \
  --label mimic500_seed111 \
  --output-dir human_study/reproduced/evaluation/mimic500_seed111
```

이 도구는 제출된 강의 `play.py`를 seed 42, 100 rollout, horizon 800, headless로 실행한다. 화면 확인은 native 명령에서 `--headless`를 생략한다. 비공개 최종 평가는 조교가 지정한 seed를 사용한다.

## 3. 저장된 결과 재집계

```bash
python human_study/scripts/summarize_comparison.py \
  --submission \
  --run-dir human_study/results/mimic500_20261028
```

시뮬레이션을 실행하지 않고 저장된 400회 판정·원본 평가 출력·학습 설정과 제출 모델 두 개를 대조해 JSON·CSV·그림을 재생성한다. Baseline 가중치와 중간 체크포인트는 제출 용량 조건에 따라 삭제했으며 설정·로그·해시를 보존했다. 이 제출 모드는 삭제된 Baseline 가중치나 HDF5 데이터를 현재 파일에서 재검증한 것으로 주장하지 않는다. 검증 범위는 `comparison.json`에 명시한다.

실행 당시 기록 JSON의 절대 경로는 당시 실험 파일의 식별 정보다. 실제 배포 모델 경로는 `models/manifest.json`과 평가 명령어를 기준으로 한다.

## 4. 데이터와 학습 재현

[데이터 안내](data/README.md)에 따라 사람 시범과 Mimic 성공 시범을 준비하고 SHA256을 확인한다. 두 원본으로 합본을 만든다. 출력 파일이 이미 존재하면 결합을 반복하지 않는다.

```bash
python human_study/scripts/prepare_mimic_training.py \
  --human human_study/data/human_20261028_112300.hdf5 \
  --mimic human_study/data/human30_mimic500_raw_20261028_env8.hdf5 \
  --output human_study/data/human30_plus_mimic500_20261028.hdf5 \
  --manifest human_study/data/rebuilt_dataset_manifest.json \
  --limit 500
```

학습 기록 도구는 강의 학습 JSON의 `train.seed`를 지정하고 종료 시 원래 설정으로 복원한다. 저장된 Baseline 설정과 비교하며 데이터·출력 경로·실험 이름 이외의 조건을 유지한다. `--dataset`에 사람 원본을 지정하면 Baseline, 합본을 지정하면 증강 조건이다. 새 학습에는 새로운 `--label`과 출력 폴더를 사용한다.

```bash
for TRAIN_SEED in 101 111; do
  python human_study/scripts/train_recorded.py \
    --seed "$TRAIN_SEED" \
    --dataset human_study/data/human30_plus_mimic500_20261028.hdf5 \
    --label "reproduce_mimic500_seed${TRAIN_SEED}" \
    --baseline-config "human_study/results/mimic500_20261028/configs/baseline_seed${TRAIN_SEED}.json" \
    --output-dir "human_study/reproduced/training/mimic500_seed${TRAIN_SEED}" || break
done
```

두 조건의 설정은 2,000 epoch, epoch당 100 update, batch 100이다. 마지막 epoch 2000을 평가한다. 새 실행의 중간 가중치를 제출 파일에 추가하지 않는다.

## 5. Mimic 생성 절차

강의 참고자료 6–8쪽과 8주차 43–44쪽의 절차를 적용했다. 다음은 증강 데이터를 다시 생성할 때의 명령이다. 이미 제공된 성공 데이터를 사용하면 이 단계는 필요하지 않다. 출력이 존재하는 경우 새 경로를 지정한다.

```bash
cd IsaacLab
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

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

`generation_guarantee=True`이므로 성공 500개를 채우는 설정이다. 실제 생성은 1,389회 시도에서 성공 500개였고, annotation은 28/30개를 통과했다. 실패 생성 시범은 학습에서 제외했다. 구현 조정과 한계는 [연구 결과](results.md), 출처 대응표는 [dataset_manifest.json](results/mimic500_20261028/dataset_manifest.json)에 있다.
