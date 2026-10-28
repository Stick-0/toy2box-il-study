# 기록·학습 설정·영상 출처 점검

**분석일: 2026년 10월 29일(KST).** 10월 28일에 완료한 실험의 로그·설정·기존 영상을 대상으로 한 사후 분석이다. 이 점검을 위해 새 모델을 학습하거나 시뮬레이션 평가를 추가하지 않았다.

## 학습 로그에서 확인되는 사실

네 run의 훈련 로그에서 epoch 1–2,000을 모두 추출했다. 기록된 `Loss`는 BC-RNN-GMM의 **행동 음의 로그우도(NLL)**이며, 매 epoch의 `Loss = -Log_Likelihood` 관계를 확인했다. 연속 확률밀도의 로그값이므로 NLL이 음수인 것 자체는 오류가 아니다.

| 학습 데이터 | 학습 seed | epoch 1 NLL | epoch 2,000 NLL | 마지막 100 epoch 평균 NLL |
|---|---:|---:|---:|---:|
| 사람 시범 30개 | 101 | −4.2347 | −51.2744 | −49.7198 |
| 사람 시범 30개 | 111 | −8.5345 | −43.4695 | −42.2661 |
| 사람 30개 + Mimic 500개 | 101 | −4.1503 | −27.8639 | −27.8180 |
| 사람 30개 + Mimic 500개 | 111 | −4.6314 | −28.2746 | −28.1490 |

Baseline도 훈련 데이터에 대한 목적함수가 개선됐다. 따라서 낮은 성공률을 단순히 “학습이 진행되지 않았다”로 설명할 근거는 없다. 다만 훈련 NLL 감소는 실제 작업 성공이나 실패 원인을 직접 측정하지 않는다. 네 run 모두 `experiment.validate=false`이고 학습 중 rollout도 꺼져 있으므로, 검증 loss 증가를 이용한 **과적합의 직접 증거는 없다.** 또한 데이터 분포가 다른 두 조건의 훈련 NLL 절댓값으로 일반화 성능의 우열을 판단할 수 없다.

전체 8,000개 epoch 값은 [training_loss.csv](training_loss.csv)에 수록했다. [evidence_audit.json](evidence_audit.json)의 `training_runs`에는 원본 로그·설정의 경로, SHA256, 계산한 요약값이 있다. Mimic 훈련 로그는 저장소 안에 있으며, baseline 원본 훈련 로그는 별도 로컬 IsaacLab_RS 프로젝트에 남아 있다. 후자는 `scope=external_local_IsaacLab_RS`로 구분한다. 저장소에는 baseline 원본 로그를 중복 복사하지 않고 추출값과 원본 SHA256을 제공한다.

## 비교 조건의 통제

동일 학습 seed의 [Baseline 설정](../results/mimic500_20261028/configs/baseline_seed101.json)과 [Mimic 설정](../results/mimic500_20261028/training/mimic500_seed101/effective_config.json)을 재귀 비교했다. 101·111 모두 차이는 다음 세 항목뿐이다.

| 항목 | 의미 |
|---|---|
| `experiment.name` | run 이름 |
| `train.data` | 사람 30개 → 사람 30개 + Mimic 500개 |
| `train.output_dir` | 학습 산출물 저장 위치 |

두 조건의 학습량은 각각 2,000 epoch × epoch당 100 update = **200,000 optimizer update**, batch size 100으로 같다. 관측·행동 표현, LSTM/GMM 구조, optimizer, 학습률, 학습 seed도 대응 조건 사이에서 같다. 111 설정을 포함한 전체 차이는 [JSON의 `training_config_differences`](evidence_audit.json)에 기록했다.

이 통제는 **Mimic 데이터 추가 효과**를 비교하는 데 적합하다. 그러나 추가된 데이터의 수와 분포가 함께 달라진다. 상태 다양성, 궤적 연결·노이즈, 생성 성공 시범 선별의 효과를 각각 분리한 실험은 아니다. 따라서 성공률 향상이 “다양성 부족만이 유일한 원인이었다”거나 “원본 시범의 수집 방식에는 아무 문제가 없었다”는 결론을 입증하지는 않는다.

## 현재 정책이 실제로 사용하는 정보

[실제 학습 설정](../results/mimic500_20261028/configs/baseline_seed101.json)의 정책 입력은 EEF 위치·자세, 그리퍼 관절 위치, 장난감 위치·자세, 박스 위치·자세다. `algo.rnn.open_loop=false`이므로 RNN은 매 step의 현재 관측으로 행동을 계산한다. 설치된 robomimic의 `BC_RNN.get_action` 구현에서도 이를 확인했다. hidden state는 설정된 horizon 10마다 초기화된다.

따라서 영상에서 빈손 운반이 나타나더라도 “현재 상태를 보지 않고 정해진 행동 순서를 그대로 재생한다”로 단정하면 안 된다. 관측에 대한 반응이 충분하지 않거나 파지 뒤와 비슷한 상태에서 후속 동작을 출력하는 현상은 가능하지만, 정확한 원인은 별도 진단이 필요하다.

환경에는 파지 여부를 계산하는 `subtask_terms.grasp`가 있지만, 해당 항목은 [학습 설정의 입력 목록](../results/mimic500_20261028/configs/baseline_seed101.json)에 포함되지 않는다. [환경 설정](../../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/place_toy2box_rmp_rel_env_cfg.py)과 [평가 코드](../../IsaacLab/scripts/imitation_learning/robomimic/play.py)에는 파지 성공을 확인해야만 운반으로 넘어가도록 정책을 제한하는 상태기계가 없다. 이 구조는 실패 후 빈손 동작을 설명할 수 있는 **대안적 가설**이며, 본 데이터 증강 실험에서 그 기여도를 분리해 검증하지는 않았다.

코드 출처·SHA256은 JSON의 `code_evidence`에 기록했다. robomimic은 저장소에 복사하지 않은 설치 의존성으로 구분하며, 확인한 심벌은 `BC_RNN.get_action`과 `BC_RNN_GMM._compute_losses`이다.

## 녹화 영상의 출처와 성공률 기록의 관계

[영상 원본 목록과 SHA256](video_source_audit.json)에 두 폴더의 MP4 200개를 기록했다. 파일 경로는 원본 IsaacLab_RS 프로젝트 기준이며, `videos/human30_seed101_eval42/`와 `videos/human30_seed111_eval42/` 각각 100개다. 200개 모두 SHA256이 달라 **파일 바이트 수준의 중복은 없다.** 해시가 다른 것만으로 영상 내용의 독립성까지 증명되는 것은 아니다.

| 구분 | 시작 시각(KST, 2026-10-28) | 영상 저장 | 용도 |
|---|---|---|---|
| Baseline 101 기존 녹화 | 16:34:54 | 있음 | 동작·실패 양상 해석 |
| Baseline 111 기존 녹화 | 17:27:02 | 있음 | 동작·실패 양상 해석 |
| Baseline 101·111 기록 평가 | 18:27:37 / 18:27:40 | 없음 | 16/100, 19/100 성공률 집계 |
| Mimic 101·111 기록 평가 | 21:14:57 / 21:14:57 | 없음 | 64/100, 60/100 성공률 집계 |

기존 영상의 Kit 실행 명령은 [baseline_audit.json](../results/mimic500_20261028/baseline_audit.json)에 보존돼 있다. 올바른 baseline checkpoint와 평가 seed 42, rollout 100회, horizon 800을 사용했고, 녹화 길이 상한은 1,000 step이었다. 영상은 나중에 성공률을 기록한 평가와 **별도 실행**이다.

이에 따라 영상의 episode 번호와 나중에 기록한 `trials.jsonl`의 trial 번호를 직접 연결하지 않는다. 같은 평가 seed라도 별도 실행의 물리 진행과 난수 소비가 같다고 보장하지 않으며, 첫 프레임·종료 시점·정책 trajectory를 연결할 기록도 없다. 영상에서 관찰한 완료와 로그의 공식 성공 판정은 구분해서 보고한다. 짧게 끝난 영상도 성공 또는 물체 낙하로 종료될 수 있으므로 **영상 길이로 성공을 판정하지 않는다.**

현재 비교 성공률의 근거는 각 평가 폴더의 `evaluation.log`, `summary.json`, `trials.jsonl`이다. 기존 `play.py`는 관측·행동 trajectory를 메모리에서 만들지만 이 평가 기록에 저장하지 않았다. 따라서 현재 자료만으로 실패 trial의 실제 3차원 오차나 접촉력을 역산해 전체 실패 원인으로 집계할 수 없다.

## 재추출

[추출 스크립트](../scripts/audit_existing_evidence.py)는 기존 파일을 읽어 위 JSON, 영상 해시 목록과 훈련 loss CSV를 생성한다. 기본적으로 저장소 옆 `IsaacLab_RS`에 원본 로그와 영상이 있는 구성을 사용한다. 원본 파일이 없는 새 clone에서는 포함된 CSV·설정·출처 해시를 검토할 수 있으며, 영상·baseline 원본 로그의 현재 무결성 재검증에는 해당 외부 원본이 필요하다.

```bash
# 저장소 루트, 강의 Python 환경에서 실행
python human_study/scripts/audit_existing_evidence.py
# 원본 프로젝트 위치가 다른 경우
python human_study/scripts/audit_existing_evidence.py --lab-root /path/to/IsaacLab_RS
```

스크립트는 환경 생성, 학습, 정책 rollout을 실행하지 않는다. 이 사후 점검은 기존 결과의 해석 범위를 명확히 하기 위한 것이며, 사전 등록된 가설이나 새로운 검증 실험으로 취급하지 않는다.
