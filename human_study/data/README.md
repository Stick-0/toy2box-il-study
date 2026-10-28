# 학습 데이터

| 구성 | 시범 수 | 스텝 수 | 제공 위치 |
|---|---:|---:|---|
| 키보드 사람 시범 | 30 | 11,769 | [human_20261028_112300.hdf5](human_20261028_112300.hdf5) |
| Mimic 성공 시범 | 500 | 210,707 | [현재 실험 데이터 Release](https://github.com/Stick-0/toy2box-il-study/releases/tag/human30-mimic500-20261028) |
| 학습 합본 | 530 | 222,476 | 위 두 파일을 제공된 결합 도구로 재구성 |

성공 생성 시범만 학습에 사용한다. 사람 원본 30개는 두 실험 조건에 동일하게 포함했다. 자동 annotation을 통과한 28개는 Mimic 생성의 소스이며, 학습에 사용한 사람 시범 수는 30개다.

Mimic 성공 데이터는 gzip 압축 시 약 134MiB여서 Release 자산으로 제공한다. [manifest.json](manifest.json)에 원본·압축 파일의 크기와 SHA256을 기록했다. 데이터 파일은 시뮬레이션 관측·행동과 로봇·물체 상태를 포함한다.

## 다운로드와 검증

저장소 루트에서 실행한다. 비공개 저장소는 접근 권한이 있는 GitHub 계정으로 `gh auth login`이 필요하다. 데이터가 이미 존재하면 다운로드와 압축 해제를 생략하고 해시만 확인한다.

```bash
gh release download human30-mimic500-20261028 \
  --repo Stick-0/toy2box-il-study \
  --pattern human30_mimic500_success.hdf5.gz \
  --dir human_study/data

printf '%s  %s\n' \
  0b62cc6a3a173ef31eeedc2e9b6ecd6e3cb6f7101fb2692f8e2bb60bd90ee4fa \
  human_study/data/human30_mimic500_success.hdf5.gz | sha256sum -c -

gzip -dc human_study/data/human30_mimic500_success.hdf5.gz \
  > human_study/data/human30_mimic500_raw_20261028_env8.hdf5
sha256sum -c human_study/data/SHA256SUMS
```

결합·학습 명령은 [재현 절차](../reproduce.md)에 있다. 원 실행에서 생성된 주석 데이터, 실패 생성 시범 889개와 합본 파일은 로컬 연구 자료로 보존한다. 실패 생성 시범은 제출 모델 학습에 포함하지 않았다.
