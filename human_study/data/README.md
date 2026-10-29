# 학습 데이터

최종 모델은 키보드로 직접 수집한 사람 시범 100개와 Mimic 성공 시범 1,000개를 합친 1,100개로 학습했다. 기존 사람 30개 및 사람 30개+Mimic 500개 실험의 데이터도 재현할 수 있도록 보존한다.

| 실험 | 구성 | 시범 수 | 스텝 수 | 제공 위치 |
|---|---|---:|---:|---|
| 사람 30개 / 사람 30개+Mimic 500개 | 사람 시범 | 30 | 11,769 | [저장소의 원본 HDF5](human_20261028_112300.hdf5) |
| 사람 30개+Mimic 500개 | Mimic 성공 시범 | 500 | 210,707 | [Human30+Mimic500 Release](https://github.com/Stick-0/toy2box-il-study/releases/tag/human30-mimic500-20261028) |
| 사람 30개+Mimic 500개 | 학습 합본 | 530 | 222,476 | 위 두 파일로 재구성 |
| 사람 100개+Mimic 1,000개 | 사람 시범 | 100 | 36,838 | [Human100+Mimic1000 Release](https://github.com/Stick-0/toy2box-il-study/releases/tag/human100-mimic1000) |
| 사람 100개+Mimic 1,000개 | Mimic 성공 시범 | 1,000 | 397,767 | 같은 Release |
| 사람 100개+Mimic 1,000개 | 학습 합본 | 1,100 | 434,605 | 위 두 파일로 재구성 |

사람 100개는 기존 30개와 추가 수집분 20개·25개·25개를 통합한 것이다. 시범 내용 단위로 원본과 대조해 중복이 없음을 확인했다. 자동 annotation은 **93/100개**가 통과했으며, 이 93개를 Mimic 생성 소스로 사용했다. 학습 합본에는 **원래 사람 시범 100개 전체**가 포함된다. Mimic은 2,680회 시도에서 얻은 성공 1,000개만 사용했고, 실패 1,680개는 학습에서 제외했다.

기존 실험은 annotation을 통과한 28/30개를 생성 소스로 사용해 1,389회 시도에서 성공 500개를 얻었으며, 학습에는 사람 30개 전체와 성공 500개를 포함했다.

## 배포 파일

큰 데이터는 Git 파일 대신 Release 자산으로 제공한다. [manifest.json](manifest.json)에 원본·압축 파일의 실제 크기와 SHA256을, [SHA256SUMS](SHA256SUMS)에 검증용 목록을 기록했다.

| 압축 파일 | 압축 크기 | 복원 파일 |
|---|---:|---|
| `human100.hdf5.gz` | 21,597,136 bytes | `human100.hdf5` |
| `human100_mimic1000_success.hdf5.gz` | 265,550,839 bytes | `human100_mimic1000_raw.hdf5` |
| `human30_mimic500_success.hdf5.gz` | 140,430,009 bytes | `human30_mimic500_raw_20261028_env8.hdf5` |

새 압축 파일은 gzip 헤더의 시각을 0으로 고정했고, 압축을 풀어 원본과 크기·SHA256이 일치하는지 확인했다. 데이터에는 시뮬레이션 관측·행동과 로봇·물체 상태가 포함된다.

## 사람 100개+Mimic 1,000개 다운로드

저장소 루트에서 실행한다. 비공개 저장소는 접근 권한이 있는 GitHub 계정으로 `gh auth login`이 필요하다. 파일이 이미 존재하면 다운로드와 압축 해제를 반복하지 않고 해시를 확인한다.

```bash
gh release download human100-mimic1000 \
  --repo Stick-0/toy2box-il-study \
  --pattern human100.hdf5.gz \
  --pattern human100_mimic1000_success.hdf5.gz \
  --dir human_study/data

sha256sum --ignore-missing -c human_study/data/SHA256SUMS

gzip -dc human_study/data/human100.hdf5.gz \
  > human_study/data/human100.hdf5
gzip -dc human_study/data/human100_mimic1000_success.hdf5.gz \
  > human_study/data/human100_mimic1000_raw.hdf5

sha256sum --ignore-missing -c human_study/data/SHA256SUMS
```

마지막 검증에서 `human100.hdf5`와 `human100_mimic1000_raw.hdf5`가 모두 `OK`인지 확인한다. `--ignore-missing`은 다른 실험의 미다운로드 파일만 건너뛴다.

## 기존 사람 30개+Mimic 500개 다운로드

사람 시범 30개 파일은 저장소에 포함되어 있다. 기존 Mimic 성공 데이터는 다음과 같이 준비한다.

```bash
gh release download human30-mimic500-20261028 \
  --repo Stick-0/toy2box-il-study \
  --pattern human30_mimic500_success.hdf5.gz \
  --dir human_study/data

sha256sum --ignore-missing -c human_study/data/SHA256SUMS

gzip -dc human_study/data/human30_mimic500_success.hdf5.gz \
  > human_study/data/human30_mimic500_raw_20261028_env8.hdf5

sha256sum --ignore-missing -c human_study/data/SHA256SUMS
```

결합·학습·증강 생성 명령은 [재현 절차](../reproduce.md)에 있다. 주석 데이터와 실패 생성 시범은 로컬 연구 자료로 보존하며 배포 학습 데이터에는 포함하지 않는다. 합본 파일은 제공된 결합 도구로 만들 수 있으므로 중복 배포하지 않는다.

## 평가 영상·상태 기록

사람 100개+Mimic 1,000개 모델의 전체 200회 평가 영상과 상태·행동 기록은 같은 Release의 `human100_mimic1000_evaluation.tar.gz`로 제공한다. 이는 평가 근거 자료이며 학습 데이터에 포함하지 않는다. 파일 구성·크기·SHA256은 [평가 아카이브 manifest](../results/human100_mimic1000/evaluation_archive.json)에 있다.

저장소 루트에서 내려받아 해시를 확인하고 별도 폴더에 압축을 푼다.

```bash
gh release download human100-mimic1000 \
  --repo Stick-0/toy2box-il-study \
  --pattern human100_mimic1000_evaluation.tar.gz \
  --dir human_study/reproduced

printf '%s  %s\n' \
  788f9f0aadb1b71d0adaca84abc68f579316bb68caaa34430258ed422b50b34e \
  human_study/reproduced/human100_mimic1000_evaluation.tar.gz | sha256sum -c -

mkdir -p human_study/reproduced/evaluation_archive
tar -xzf human_study/reproduced/human100_mimic1000_evaluation.tar.gz \
  -C human_study/reproduced/evaluation_archive
```
