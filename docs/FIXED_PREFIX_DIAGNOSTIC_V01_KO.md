# 고정 prefix 좌표·logit 진단 v01 — 수집 완료·결과 검토

**2026-09-30 사용자 [실행 승인](FIXED_PREFIX_APPROVAL_20260930_KO.md) 후 12회 완료 run.json을 받아 [결과 검토](FIXED_PREFIX_RESULT_REVIEW_20260930_KO.md)를 기록했다. 이후 [저장 logits의 CPU 감사](LOGITS_AUDIT_REVIEW_20260930_KO.md)도 수신·검토 완료했다. GPU 진단이나 같은 CPU 감사를 다시 실행하지 않는다.** 기존 R0 승인 및 완료한 자체20문제 시험과 구분한다. 이 문서는 과거 산수2개의12회 forward 후보를 현재 D2·D3 입력으로 구체화한 새 범위다.

관련 파일: [설정](../configs/fixed_prefix_diagnostic_v01.json), [계약·수치 집계](../scripts/fixed_prefix_diagnostic_contracts.py), [실행기](../scripts/run_fixed_prefix_diagnostic.py), [CPU 마스크 검토](SDPA_MASK_REVIEW_20260929_KO.md).

## 목적과 고정 입력

질문은 스케일 적용·역좌표 변환·norm/bias 잔차가 **동일 prefix의 다음 토큰 분포를 얼마나 바꾸는가**다. 자유 생성으로 난도 시험을 반복하거나 검열된 답을 완성하는 실험이 아니다.

| 원본 응답 | 원본 입력 | 재사용할 생성 앞부분 | 전체 입력 | logit 위치 |
|---|---:|---:|---:|---|
| D2-03 / BF16 / seed42 |71|2048|2119|70,86,…,2118의129개|
| D3-01 / BF16 / seed42 |79|2048|2127|78,94,…,2126의129개|

첫 위치는 첫 생성 토큰을 예측하는 prompt 마지막 위치다. 이후 16토큰 간격이며 마지막은 저장된 2048토큰 뒤의 다음 토큰을 예측한다. 마지막 위치까지 causal context를 전부 계산하고 해당129개 위치의 전체 vocabulary logit을 저장한다. 두 입력에 EOS는 없다. 입력을 다시 토큰화하지 않는다.

원본 행과 ID 배열의 해시를 고정했다. 기존 결과를 본 뒤 선택한 탐색 표본이다. D2-03 seed42는 BF16 EOS/AWQ 검열, D3-01 seed42는 양쪽 검열이었다. 두 BF16 경로를 모든 상태에 동일하게 공급하므로 AWQ 자체 생성 경로의 대표성이나 독립 확인 표본을 주장할 수 없다. MATH-500·새 Pile 자료는 사용하지 않는다.

## 상태·계산 예산

| 상태 | 구성 | 비교 의미 |
|---|---|---|
| B, B_repeat | 같은 고정 BF16, 동일 입력 각각1회 | 반복 결과 일치 여부. 다르면 후속 상태 전에 중단 |
| S | B에 R0의 저장 scale만 적용 | B↔S: 양자화 전 재매개화 영향 |
| Q | B를 다시 로드한 뒤 기존 scale/clip/W3 g128 재적용 | B↔Q: 양자화 효과 크기. 같아야 하는 비교가 아님 |
| C | Q를 float64 원래 좌표로 역변환한 뒤 GPU BF16 파라미터로 복사 | Q↔C: 좌표 복원 영향 |
| Cw | C의 선형 weight를 유지하고 원래 B의 norm/bias를 복사 | C↔Cw: 양자화 대상 밖 잔차의 영향. 주 AWQ 조건을 자동 교체하지 않음 |

최대 **2입력×6회=12회 full-sequence forward**, 입력 처리 합계 **25,476토큰**, 최대 문맥 **2,127토큰**, **새 생성 0토큰**이다. B 반복이 비동일하면4회에서 중단한다. 모델 로딩·파라미터 해시·28계층 변환/재구성·좌표 복원 비용은 forward 횟수와 별도다. 준비 시점에는 시간·GPU peak를 측정하지 않았고 이후 실행 보고값은 결과 검토에 기록했다. 무료 계산이라는 의미가 아니다.

가중치는 한 모델씩 유지한다. 비교를 위해 BF16 logit을 FP32로 바꿔 12개의 `.npy`에 저장한다. 데이터 본체 최대940,787,712바이트(약897MiB)와 JSON/배열 헤더 공간이 필요하다. 모델/데이터를 다운로드하거나 AWQ calibration을 다시 탐색하지 않는다. 오류·중단·기존 출력 폴더 발견 시 자동 재시도/재개하지 않는다.

## runtime과 근거 보호

- 기존 1.5B 모델 revision, torch2.7.1+cu118 / Transformers4.51.3 / Hub0.36.2 / NumPy1.26.4, cuda:0, BF16, SDPA를 고정한다. TF32는 끄고 float32 matmul precision은 highest, deterministic_algorithms는 false로 명시한다.
- [마스크 검토](SDPA_MASK_REVIEW_20260929_KO.md)를 반영해 진단용 설정 복사본의 `sliding_window=None`을 모델 생성 전에 적용한다. `use_sliding_window=false`는 유지한다. 원본/effective 설정 및 실제 model.config 해시를 각각 남긴다. 패키지 소스나 원래 config.json은 수정하지 않는다.
- batch1, `use_cache=false`, 명시적 attention/position/cache_position, attention 출력 없음. 최대문맥이 경계 이전인 이 설계는4096 전후 마스크 영향이나 생성 KV-cache 경로를 검증하지 않는다.
- 수신 근거2개, 고정 입력, 원 R0의29개 파일, AWQ 소스6개와 Transformers 소스4개 해시를 확인한다. B와 재구성 Q의 전체 파라미터 해시가 기존 난도 시험의 기록과 다르면 각 상태의 forward 전에 중단한다. 원 R0에는 전체 AWQ 파라미터 해시가 없어 그 체크포인트와의 동일성을 인증하지 않는다.
- C의140개 norm/bias에서 float64 역변환 배열과 B, BF16 재적용 값과 B, 재적용 전후를 구분한다. 참조 RMS·오차 RMS·상대 RMS·max_abs와 위치를 기록한다. 참조 RMS=0이면 상대값은 null이다. C→Cw에서는196개 선형 weight의 해시가 변하지 않았는지 확인한다.

## 판정 규칙

1. **수집 실패/미완료:** 근거·파라미터·예산·shape 불일치, 비유한 값, 누락, 예외. 후속 생성/매칭으로 자동 진행하지 않는다.
2. **반복 점검 보류:** B와 B_repeat의129위치×전체 vocabulary 값이 정확히 일치하지 않으면4회에서 중단하고 runtime 변동을 먼저 검토한다. 이 중단은 양자화 실패 판정이 아니다.
3. **자료 수집 완료:** 최대12회 및 입력별5비교가 모두 기록되면 `FIXED_PREFIX_DIAGNOSTICS_COLLECTED_PENDING_REVIEW`. 전체 모델 동등성 PASS가 아니다.
4. **비교별 분류:** 정확한 logit 일치 / top-1 변경 / logit 변경이 있으나 top-1 동일을 구분한다. top-1이 같아도 sampling 분포가 같다고 하지 않는다. logit의 공통 상수 이동은 분포를 바꾸지 않으므로 분포 지표도 함께 읽는다.

logit RMS·상대 RMS·max_abs·최대 위치, top-1/2 margin과 flip, KL(Baseline→Compared)·TV, EOS log-probability 및 확률 차이를 위치별로 저장한다. 분포는 T=1·top-p 전·전체 vocabulary의 **NumPy float64 log_softmax**로 계산한다. 이는 옛 FP32 집계 초안과 구분되는 이 버전의 명시적 선택이다. 실제 sampling의 T=0.6/top-p=.95 필터까지 검증하지 않으며 KL 매칭 실험도 아니다.

top-1 margin이 `2×max_abs_logit_delta`보다 큰데 argmax가 달라지는 것은 수학적 경계와 모순되므로 구현 오류로 중단한다. 이 검사는 기능적 허용오차를 대신하지 않는다. 정확히 같다는 판정도 선택된 위치에 한정한다. 비영 차이를 기존R0 RMS나 합성5%에 맞춰 소급 PASS로 처리하지 않는다. 실용적 동등성 한도·다른 prefix·cache 경로까지의 수락 기준은 아직 미정이며 이번 진단 후 별도 독립 점검 전에 정해야 한다. 항상 model_ready=false를 남긴다.

## 현재 상태와 승인된 별도 후속

`authorization=APPROVED`는 역사적 승인 기록으로 유지하며 승인된12회는 이미 수집됐다. 실행 commit907ca61의 코드/설정/연구 상태 해시가 일치한다. BF16 반복0/258, B→Q22/258, Q→C3/258의 top-1 변경을 검토했다. 사용자 로컬의12개 배열·10개 비교 감사 보고도 일치했다. 다음은 [종료 직전 경로 추가6회 진단](TERMINATION_PREFIX_V01_KO.md)이다. 2026-09-30 사용자 “진행해줘”를 [제안된 범위의 실행 승인](TERMINATION_PREFIX_APPROVAL_20260930_KO.md)으로 기록했다. 아래 명령은 사용자 GPU에서 새 진단을 1회 실행한다. 기존12회를 다시 실행하거나 같은 승인을 다시 요청하지 않는다.

```bash
conda activate quantthink
cd ~/quantthink
git switch setup/session-01-research-gates &&
git pull --ff-only &&
python scripts/run_termination_prefix_diagnostic.py --execute
```

새 실행 결과 `results/local/termination_prefix_v01/run.json`을 검토한다. 사용자 GPU 실행 결과는 아직 받지 않았다. 기존12회 실행기의 `--execute`·`--plan` 출력 재제출은 필요 없다. [CPU 감사의 수치 정의·동점 처리·판정](FIXED_PREFIX_RESULT_REVIEW_20260930_KO.md)을 따른다.

수신 run.json과 `results/local/fixed_prefix_v01/logits_audit_v01.json` 검토는 완료했다. 계획 조회도 실제 첨부에서 확인했으므로 재제출은 필요 없다. 약0.94GB logit 배열은 로컬에서 보존한다. 기존20문제·R0 결과와 별도 폴더이며 원본 결과/원시배열을 공개 Git에 올리지 않는다. 기존 출력 폴더가 있거나 중단되면 결과를 지우거나 자동 반복하지 말고 현재 run.json과 로그를 검토한다.

정상 수집 상태는 `FIXED_PREFIX_DIAGNOSTICS_COLLECTED_PENDING_REVIEW`다. `BASELINE_REPEAT_DIFFERENCE_REQUIRES_REVIEW`이면 BF16 반복4회 결과부터 검토한다. 어느 상태도 실모델 동등성 PASS나 세션03 전환을 뜻하지 않는다. 이 대화 환경에서 사용자 GPU 실행을 시작한 것은 아니다.

## 작성 환경에서 확인한 범위

새 CPU 검사7개가 입력/위치·근거 해시 보호, 새로운 실행 범위 gate, 모델 import 없는 계획 조회, 알려진 KL/TV/EOS 분포, logit 상수 이동과 argmax, 비유한 값·shape 오류·0인 참조 RMS를 다뤘다. 준비 시점에 실제 첨부2개로 prefix/예산을 대조하고 당시 미승인 `--execute`의 조기 거부도 확인했다. 승인 기록 갱신에서는 코드/설정이 그대로인지와 순수 Python 승인 조건만 대조했다. 고정 Qwen2 소스가 tensor형 `logits_to_keep`와 명시적 위치 인자를 받는지 대조했다.

**PyTorch/Transformers가 없는 작성 환경이므로 새 실행기의 실모델/CUDA 경로는 검증하지 못했다.** 기존 준비·토크나이저·합성 AWQ·R0 검사는 반복하지 않았다. 이 절은 작성 당시의 검사 기록이다. 이후 사용자12회 수집 보고를 검토했으며 원시 logit 감사·기능적 수락 판단은 별도로 남아 있다.
