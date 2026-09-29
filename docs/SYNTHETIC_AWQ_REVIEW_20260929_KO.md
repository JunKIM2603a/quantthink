# 합성 AWQ 점검 검토 및 데이터 준비 — 2026-09-29

검토 당시 세션: **QuantThink 01 — 저장소 점검·연구 설계 검증**  
대조한 저장소 commit: `4076739762357b3e26b7b89641bb96f131b07fff`

**사용자 제출 결과는 작은 무작위 Qwen2의 AWQ 연결 점검을 통과했다. 이후 고정 데이터 준비도 완료 보고를 검토했다.** 실제 사전학습 모델의 BF16/AWQ 재현, 장문맥, H1/H2 결과는 아직 없다.

후속 행 길이 오류는 [수정 기록](DATA_PREPARATION_FIX_20260929_KO.md), 이후 성공 보고는 [데이터 준비 완료 검토](DATA_PREPARATION_REVIEW_20260929_KO.md)에 정리했다. 이후 [R0 검토안](R0_REVIEW_PACKET_20260929_KO.md)의 사용자 승인으로 세션 02로 전환했다. 현재 실행 안내는 [인계문](HANDOFF_01_TO_02_KO.md)에 있다. 이 문서의 합성 AWQ 해시 대조는 위 commit 기준의 과거 근거이며 아래 데이터 준비·요약 명령을 다시 실행할 필요는 없다.

## 1. 확인한 근거와 범위

| 항목 | 제출 결과 및 검토 |
|---|---|
| 단위 테스트 요약 | 32개 중 31개 통과·1개 건너뜀, 실패 없음. 개별 검사 이름과 건너뛴 사유는 제출되지 않음 |
| AWQ 원본 준비 | 모듈 5개와 LICENSE, 총 6개 해시 일치라는 사용자 출력 |
| GPU 연결 상태 | `SYNTHETIC_AWQ_ADAPTER_PASS_NOT_MODEL_READY` |
| 실행 환경 | torch 2.7.1+cu118, transformers 4.51.3, numpy 1.26.4, tqdm 4.70.1; cuda:0 |
| 검사 모델·입력 | 고정 정책 기준: 무작위 1계층 Qwen2, hidden 128, intermediate 256, vocabulary 256, seed 0, 입력 1 × 512토큰 |
| 구현 해시 | 제출된 Python 파일 4개의 SHA-256이 저장소 바이트와 모두 일치 |
| 정책 해시 | 제출된 정책 SHA-256이 저장소 JSON의 정규화 해시와 일치 |
| 점검 시간 | 사용자 보고 약 1.66초. 실모델 처리량 또는 전체 실행 시간 추정치로 사용하지 않음 |

자료는 사용자가 붙여 넣은 터미널 출력이다. 원본 로컬 JSON 바이트와 보고서를 생성한 Git commit은 받지 않았으며, GPU 검사를 이 작성 환경에서 재실행하지 않았다. 대조 대상 commit의 원격 38개 파일과 로컬 검토 사본의 Git blob 해시를 모두 확인했다.

“코드는 아직 실행하지 않았습니다”는 첫 번째 **원본 다운로드 명령의 종료 시점**을 설명한다. 이후 합성 어댑터 검사에서는 고정 원본의 스케일 탐색·clipping·W3 변환 함수가 실행된다. 두 메시지는 서로 모순되지 않는다.

단위 검사 소스에는 zstandard가 없을 때 건너뛰는 압축 JSONL 검사와, PyTorch·Transformers가 없을 때 건너뛰는 Qwen2 검사 3개가 있다. 제출된 요약만으로 건너뛴 1개를 확정하지 않는다. 후속 데이터 준비에서는 zstandard 0.23.0 설치와 실제 압축 파일 읽기를 확인했다.

## 2. 계층 출력 차이의 해석

| 검사 | 최대 절대 오차 | RMS | 상대 RMS | 백분율 |
|---|---:|---:|---:|---:|
| 스케일만 적용한 출력 차이 | 0.0009765625 | 0.00008343348816311949 | 0.0037016145199817822 | 약 0.3702% |
| 원래 좌표로 되돌린 출력 차이 | 0.0009765625 | 0.00008227643656872291 | 0.0036649665056165473 | 약 0.3665% |

두 상대 RMS 모두 사전에 정한 합성 점검 한도 **0.05 = 5%** 이내다. 이 한도는 소형 무작위 모델에서 큰 연결 오류를 찾는 기준이다. `canonicalized_output_difference`는 역변환한 양자화 모델과 재매개화 상태의 양자화 모델 사이의 비교이며, 원래 BF16 모델에 대한 양자화 정확도 손실이 아니다. 실제 모델의 함수 동등성이나 H1의 동등성 경계를 인증하지 않는다.

## 3. SDPA 경고 검토

Transformers **v4.51.3**의 공식 Qwen2 설정·구현을 직접 읽었다.

- 합성 검사에서 생략한 기본값은 `use_sliding_window=False`, `sliding_window=4096`, `max_window_layers=28`이다.
- 계층 생성자의 경고 조건은 `config.sliding_window`가 참이고 구현이 `flash_attention_2`가 아닌지 확인한다. 경고 조건에는 `use_sliding_window` 검사가 없다. 따라서 현재 기본 설정과 SDPA 조합에서도 제출된 문구가 발생한다.
- attention에 넘기는 창 크기는 `use_sliding_window`와 계층 조건을 확인한다. 다만 같은 버전의 인과 마스크 경로에는 `config.sliding_window`를 직접 확인하는 분기도 있다. `use_sliding_window=False`만으로 모든 길이에서 영향을 배제하면 안 된다.
- 이번 입력은 512토큰이고 cache를 사용하지 않는다. 창 경계는 4,096토큰이므로 이 입력 범위에서는 창으로 인해 과거 토큰이 추가로 가려지지 않는다. **이 경고만으로 이번 합성 PASS를 기각할 근거는 없다.** 긴 문맥에서의 정확성은 별도 검증 항목이다.

경고를 없애기 위해 attention backend나 패키지를 바꾸지 않는다. 실모델의 실제 checkpoint 설정, 긴 입력의 mask, KV cache 동작은 해당 실행 전에 따로 검토한다. 이번 결과는 40,960토큰 후보 문맥의 검증이 아니다.

검토한 1차 소스:

- [Qwen2 설정 v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/configuration_qwen2.py), Git blob `2e82f1976f3922f3620415f4eace6c6e046243f8`.
- [Qwen2 구현 v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py), Git blob `16a7316e2d0e56eafe301a7f2d8693d6cc6c73ec`. 경고·attention 인자·인과 마스크 분기를 확인했다.

## 4. 완료한 개발 데이터와 calibration 후보 준비 명령

이후 실행 명령에는 conda 활성화를 포함한다. 사용자가 적은 이름은 `quantrhink`이지만 제출된 활성 환경 표시는 `(quantthink)`이므로 아래에서는 **`quantthink`**을 사용한다. 환경을 새로 만들거나 이름을 바꾸지 않는다.

아래는 당시 전달한 준비 명령이다. `--mode all`로 개발 데이터와 calibration의 준비 완료를 확인했으므로 데이터 준비·토크나이저·합성 GPU 검사를 반복할 필요는 없다.

~~~bash
conda activate quantthink
cd ~/quantthink
git switch setup/session-01-research-gates
git pull --ff-only

python -m pip install --no-deps -r requirements/data-preparation.txt

python scripts/prepare_runtime_assets.py \
  --mode all --online \
  --output-dir results/local/runtime_assets_v03
~~~

준비용 추가 패키지는 pyarrow 20.0.0과 zstandard 0.23.0이다. 고정된 Hugging Face Hub 0.36.2 및 Transformers 4.51.3도 검사한다. 오류가 나면 마지막 오류 문구를 공유하고, 해시·revision·버전을 임의로 바꿔 통과시키지 않는다. 같은 출력 폴더가 이미 있으면 덮어쓰지 않으므로 새 실행에서는 다른 경로를 지정하고 아래 조회 경로도 맞춘다.

| 자산 | 이번 작업 |
|---|---|
| GSM8K main/train | 고정 파일 해시와 7,473행 확인, 개발 문제 100개 선택 |
| MATH-500 | 고정 파일 해시와 500개 고유 문제 확인, 문제 텍스트 해시로 중복 제외. 정답 평가 없음 |
| Pile-uncopyrighted validation | 고정 압축 파일 해시 확인, 적격 문서 후보에서 정확히 128 × 512 = 65,536토큰 준비 |
| 토크나이저·설정 | 이미 검토한 revision과 설정 해시를 대조해 calibration 문서를 토큰화 |

Pile 파일은 앞선 공개 목록에서 약 338MB였다. 실제 다운로드 합계는 실행 시 고정 revision의 메타데이터로 표시하고 512MiB 상한을 검사한다. 다운로드 후 전체 calibration 후보를 CPU에서 토큰화하므로 시간이 걸릴 수 있다. 모델 가중치 로딩이나 생성은 수행하지 않는다.

성공 상태는 **`PREPARED_CANDIDATE_NOT_RUN_APPROVAL`**이다. 이는 출처가 확인된 후보 파일 준비의 완료이며 프로토콜 수락·연구 실행 승인과는 별개다.

## 5. 제출 완료된 결과의 요약 명령 기록

아래 요약 출력은 이미 제출·검토 완료됐다. 이후 로컬에서 결과를 조회할 때 사용할 수 있다. 원문 문제·정답·문서와 토큰 배열을 출력하지 않고 출처·해시·선택 개수만 모은다. 이 명령은 보고서를 읽는 요약 명령이며 독립적인 파일 무결성 검사를 대신하지 않는다.

~~~bash
conda activate quantthink
cd ~/quantthink
python - <<'PY'
import json
from pathlib import Path

root = Path("results/local/runtime_assets_v03")
report = json.loads((root / "preparation_report.json").read_text())
dev = json.loads((root / "development_manifest.json").read_text())
cal = json.loads((root / "calibration_manifest.json").read_text())
summary = {
    "preparation_report": report,
    "development_summary": {
        key: dev[key] for key in (
            "status", "counts", "selection_seed", "selected_ids_sha256",
            "confirmation_question_set_sha256"
        )
    },
    "calibration_summary": {
        key: cal[key] for key in (
            "status", "counts", "total_tokens", "used_documents",
            "token_blocks_sha256", "tokenizer", "jsonl_reader"
        )
    },
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
~~~

선택 ID와 사용 문서의 상세 목록은 로컬 manifest에 남긴다. 원문 데이터·정답·토큰 배열은 공개 저장소에 올리지 않는다. 이후 검토에서 출처 3개, 실제 코드·설정 해시, 개발 100개, calibration 65,536토큰, 제외·중복 개수와 선택 해시를 대조했다. 상세 근거 수준은 최신 완료 검토 문서를 따른다.

## 6. 연구 단계 판정

검토 당시 세션 01을 유지했고, 이후 데이터 준비 보고 검토와 제안 기여 범위·R0 프로토콜의 사용자 승인을 기록해 세션 02로 전환했다. 다음 작업은 사전학습 모델의 R0 진단이다. BF16/AWQ 재현·KL 매칭·H1/H2는 계속 미실행이다.

이 합성 AWQ 검토 당시 갱신은 문서와 상태 기록이었으며 구현 파일 및 설정 JSON 3개는 변경하지 않았다. 이후 데이터 준비 수정에서는 읽기 코드가 바뀌었다. 변경 범위와 검증은 별도 수정 기록을 참조한다. 이미 받은 GPU 검사 재실행을 요청하지 않는다. 문서 링크·명령 구문·상태 일관성과 코드·설정 해시를 확인하며, 이 작업을 새로운 모델 실험이나 단위 테스트 통과로 합산하지 않는다.
