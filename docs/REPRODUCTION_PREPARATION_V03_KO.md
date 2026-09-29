# QuantThink 재현 준비 v0.3 — 실제 AWQ 연결과 출처 검증

작성일: 2026-09-29  
현재 세션: **QuantThink 01 — 저장소 점검·연구 설계 검증**

이번에는 실제 원본 AWQ 함수를 호출하는 Qwen2 어댑터, 데이터 파일의 출처·바이트 검증 로더, R0 실행기 후보를 구현했다. **구현을 완료한 범위와 실제로 실행해 확인한 범위는 아래와 같이 다르다.** 현재 환경에는 PyTorch·Transformers가 없어 실모델이나 GPU 어댑터를 실행하지 않았다.

## 1. 이번 결과와 증거

| 항목 | 이번 상태 |
|---|---|
| 고정 원본 AWQ 함수 연결 | 구현. 실제 CUDA 실행은 아직 미검증 |
| 고정 revision의 데이터 로더 | 구현. 모의 Hub 응답과 합성 파일로 검증. 실제 원격 데이터 준비는 미실행 |
| Qwen2 입력 포착·스케일·clip·W3 적용 | 코드 작성 및 구문 검사. 사용자 GPU에서 합성 점검 필요 |
| R0 입력·EOS·종료·자산 연결 | 구현. 현재 승인 기록으로는 사전학습 모델 실행 전에 중단 |
| 새 v0.3 테스트 | 32개 중 29개 통과, PyTorch·Transformers가 필요한 3개는 건너뜀 |
| 기존 v0.2 연계 테스트 | 32개 재실행·통과 |
| 실제 데이터 ID·AWQ 모델 결과·장문맥·H1/H2 | 아직 없음 |

검사 환경은 Python 3.12.14, NumPy 2.3.5, zstandard 0.25.0이다. 압축 JSONL의 여러 프레임을 읽는 합성 검사는 실제 zstandard로 수행했다. PyArrow의 실제 Parquet 읽기와 Hub 다운로드는 이 환경에서 실행하지 않았다. 사용자용 데이터 준비 버전은 pyarrow 20.0.0, zstandard 0.23.0 후보로 고정했으므로 해당 버전의 실제 실행 결과도 추가로 받아야 한다.

## 2. 지금 사용자가 실행할 순서

기존 quantthink 환경을 사용한다. 현재 검토 대상은 PyTorch 2.7.1+cu118, Transformers 4.51.3이며, 이번 점검 때문에 PyTorch를 다시 설치하지 않는다.

~~~bash
cd ~/quantthink
git switch setup/session-01-research-gates
git pull --ff-only

python -m unittest discover -s tests -p 'test_runtime_preparation_v03.py' -v

python scripts/prepare_runtime_assets.py \
  --mode upstream --online \
  --output-dir results/local/awq_reference_v03

python scripts/check_awq_adapter.py \
  --upstream-dir results/local/awq_reference_v03 \
  --device cuda:0 \
  --output results/local/awq_adapter_check_v03.json
~~~

단위 테스트에서는 사용자 환경의 PyTorch·Transformers를 이용해 앞서 건너뛴 3개의 Qwen2 CPU 연결 검사도 수행할 수 있다. 검사가 실패하면 오류를 해결한 뒤 다음 명령으로 넘어간다.

upstream 준비 명령은 고정한 Python 모듈 5개와 LICENSE만 내려받아 Git blob 해시를 검사한다. 합계 약 25KB이며 모델·데이터셋을 받지 않는다. 이미 일치하는 파일은 재사용하고 중단으로 누락된 파일은 이어 받는다. 해시가 다른 기존 파일은 덮어쓰지 않는다.

합성 점검은 **사전학습 가중치와 벤치마크 데이터를 사용하지 않는 작은 무작위 Qwen2**를 GPU 한 장에서 실행한다. 설정은 hidden 128, intermediate 256, 계층 1개, vocabulary 256, 무작위 입력 1 × 512 토큰, seed 0이다. 실제 원본의 스케일 탐색·clipping·W3 가짜 양자화를 호출하므로 단순한 NumPy 계산보다 한 단계 진행된 연결 검사다.

성공 기대 상태는 SYNTHETIC_AWQ_ADAPTER_PASS_NOT_MODEL_READY이다. 콘솔에 상태, 패키지 버전, 코드·설정 해시, 계층 출력 오차가 함께 표시된다. **이 콘솔 출력을 다음 검토 자료로 사용한다.** 상세 스케일·clip 값은 로컬 JSON에 보관한다. 기존 토크나이저 검사는 반복할 필요가 없다.

합성 점검의 상대 RMS 한도 0.05는 큰 구현 오류를 잡기 위한 느슨한 소형 모델 검사 기준이다. 실제 BF16 모델의 함수 동등성이나 H1의 효과 크기 경계로 사용하지 않는다.

## 3. 추가한 파일

| 파일 | 역할 |
|---|---|
| [runtime_preparation_v03.json](../configs/runtime_preparation_v03.json) | 원본 경로·소스 해시·calibration 선택·합성 검사 후보 설정 |
| [runtime_assets.py](../scripts/runtime_assets.py) | 고정 출처 계획, 파일 바이트 검증, 문제·calibration 선택 |
| [prepare_runtime_assets.py](../scripts/prepare_runtime_assets.py) | 오프라인 점검 및 명시적인 온라인 준비 명령 |
| [qwen2_awq_adapter.py](../scripts/qwen2_awq_adapter.py) | 원본 AWQ 탐색·변환과 Qwen2 계층 실행 연결 |
| [check_awq_adapter.py](../scripts/check_awq_adapter.py) | 작은 무작위 모델의 CUDA 연결 점검 |
| [runtime_contracts.py](../scripts/runtime_contracts.py) | R0 입력·종료·승인 상태·준비 파일 대조 |
| [run_r0.py](../scripts/run_r0.py) | 승인 조건 검사와 BF16/AWQ R0 실행기 후보 |
| [test_runtime_preparation_v03.py](../tests/test_runtime_preparation_v03.py) | 변조·부분 파일·데이터 분리·실행 누락·연결 검사 |
| [data-preparation.txt](../requirements/data-preparation.txt) | 데이터 준비용 추가 패키지 두 개 |

기존 reproduction_candidate.json과 asset_inspection_refs.json은 변경하지 않았다. v0.3은 별도의 구현 후보이며 원래 H1/H2와 연구 승인 상태를 바꾸지 않는다. 이전 문서는 [v0.2 기록](REPRODUCTION_PREPARATION_V02_KO.md)에 남아 있다.

## 4. 실제 AWQ 어댑터의 처리

원본은 ruikangliu/Quantized-Reasoning-Models의 bf947e29f52e3f666e3263efac149dae0ac18d00이다. 실행 전에 필요한 원본 모듈의 Git blob 해시를 다시 검사하고, 별도 모듈 이름으로 가져온다. 원본 저장소의 전체 학습·데이터 스크립트를 실행하지 않는다.

어댑터는 사용자가 지정한 CUDA 장치를 기본 장치로 설정해 원본 내부의 .cuda() 호출도 같은 worker에서 처리한다. Qwen2ForCausalLM, BF16 파라미터, 검토한 7개 선형 계층, g128과 clipping 채널 조건이 맞지 않으면 중단한다.

첫 계층의 입력과 kwargs를 forward pre-hook으로 포착한다. position_embeddings, attention mask, position 관련 인자를 보존하며 KV cache는 사용하지 않는다. 포착용 전용 예외만 처리하고 실제 ValueError 등을 삼키지 않는다. hook과 use_cache 설정은 종료·실패 경로에서도 정리한다.

각 계층에서 다음을 수행한다.

1. 원래 계층의 입력·출력과 각 선형 계층의 입력을 수집한다.
2. 고정 원본의 스케일 탐색·적용을 수행하고 양자화 전 출력 차이를 기록한다.
3. 원본의 clipping 탐색·적용 뒤, 7개 선형 weight를 W3·g128·비대칭 방식으로 가짜 양자화한다.
4. 스케일 기록을 역순으로 되돌려 동일 BF16 참조 좌표에서 가중치 오차를 계산한다.
5. 복원한 가중치를 일시 적용해 출력 차이를 측정한 뒤, 실제 평가할 AWQ 상태를 복구한다.

다음 계층 탐색에는 **변환하기 전 참조 계층의 출력**을 전파한다. 이는 읽은 원본 탐색 순서와 같은 선택이다. 양자화 출력으로 다음 탐색 입력을 바꾸는 방식은 이번 구현에 섞지 않았다.

양자화 대상 밖의 norm·bias에도 BF16 스케일 연산의 반올림 잔차가 남을 수 있다. 이를 canonical_auxiliary_error에 별도로 기록한다. 후속 가우시안 대조군에서 어떤 파라미터를 교란할지는 이 잔차와 함께 검토해야 하며, 선형 weight 오차만 맞췄다고 모든 교란이 같아졌다고 보지 않는다.

반환 상태는 AWQ_TRANSFORM_APPLIED_NOT_EQUIVALENCE_CERTIFIED이다. 계층 출력 오차를 기록했다는 사실이 전체 모델 logit 동등성이나 실제 checkpoint의 수락 기준 통과를 뜻하지 않는다. 변환 도중 실패한 부분 모델은 다시 사용하지 않는다.

## 5. 데이터 출처 검증과 새 calibration 후보

Hugging Face 공식 endpoint에 전체 40자리 revision을 지정하고 files_metadata=True로 크기·해시를 받는다. 응답 revision이 다르거나 메타데이터가 없으면 중단한다. 토큰을 보내지 않는 공개 다운로드를 사용한다.

| 역할 | 원본 경로 | 확인 방식 |
|---|---|---|
| GSM8K 개발 | main/train-00000-of-00001.parquet | 파일 해시, main/train 예상 7,473행, question·answer 필드 |
| MATH-500 확증 | test.jsonl | 파일 해시, 500개 고유 문제. 선택·출력에는 problem 필드만 사용 |
| Pile calibration | val.jsonl.zst | 고정 revision의 압축 파일 전체 해시를 먼저 확인한 뒤 읽음 |

일반 Git 파일은 blob 헤더를 포함한 SHA-1, LFS 파일은 실제 내용의 SHA-256을 대조한다. LFS pointer의 Git 해시를 내용 해시로 잘못 쓰지 않는다. 별도로 모든 내려받은 파일의 SHA-256과 실행 코드 해시도 보고서에 남긴다.

이 검증은 공식 Hub의 고정 revision 메타데이터와 파일 바이트의 일치 확인이다. 데이터 내용의 정답성·의미 중복·사전학습 오염 부재까지 인증하지 않는다. MATH-500 원본 파일에는 정답도 포함되지만, 준비 코드는 원본 바이트 검증 후 문제 문자열만 보존·사용한다. 확증 정답을 튜닝하거나 평가하지 않는다.

이번 calibration 구현 후보는 다음과 같다.

- Pile-uncopyrighted의 validation 파일을 사용한다. 원래 후보에서 미정이던 split을 이번 구현 후보에 명시했으며 원 논문의 정확한 표본을 재현했다고 주장하지 않는다.
- text.strip() 이후 길이가 1~512 토큰인 문서를 대상으로 한다. 특수 토큰을 추가하지 않는다.
- GSM8K train·MATH-500 문제와 전체 정규화 텍스트가 같은 문서 및 반복 문서를 제외한다. 반복 문서는 원본 순서의 첫 항목을 유지한다.
- seed 42와 원본 행 번호의 SHA-256이 작은 적격 문서 1,024개를 후보 풀로 만든다.
- 그 순서로 구분 토큰 없이 연결해 정확히 128 × 512 = 65,536 토큰을 사용한다. 마지막 문서를 일부만 사용하면 그 길이도 기록한다.
- 부족한 경우 재표집·split 교체·예산 변경 없이 중단한다.

문서 ID는 고정 파일의 행 번호와 텍스트 해시로 남긴다. 사용한 토큰 해시, 전체 블록 해시, 제외 개수와 전처리도 기록한다. 코퍼스 중복의 모든 형태를 잡는 절차는 아니다. 남은 GSM8K 문제를 KL 매칭·검증에 자동 배정하지 않는다.

공개 파일 목록에서 Pile validation 파일은 약 338MB로 표시된다. 실제 다운로드 크기는 고정 revision의 응답으로 다시 검사하며, 데이터 파일 합계 상한은 512MiB이다. 숫자를 넘으면 추가 다운로드를 진행하지 않는다. 선택한 원본 경로가 고정 revision에 실제로 존재하는지와 실제 데이터 바이트 일치는 사용자 측 온라인 실행에서 최종 확인한다.

## 6. 합성 연결 점검 이후의 데이터 준비

개발 데이터 준비 또는 전체 calibration 준비가 필요할 때만 다음 명령을 사용한다. 현재 설치된 Torch·NumPy·Transformers·Hugging Face Hub를 일괄 갱신하지 않는다.

~~~bash
python -m pip install --no-deps -r requirements/data-preparation.txt

# 개발 문제 목록과 정답 파일만 준비
python scripts/prepare_runtime_assets.py \
  --mode development --online \
  --output-dir results/local/development_assets_v03

# 이후 calibration까지 준비할 때: 별도의 새 디렉터리 사용
python scripts/prepare_runtime_assets.py \
  --mode all --online \
  --output-dir results/local/runtime_assets_v03
~~~

전체 준비에는 토크나이저가 필요하지만 모델 가중치나 생성은 사용하지 않는다. 성공 상태는 PREPARED_CANDIDATE_NOT_RUN_APPROVAL이다. 고정한 파일의 출처가 확인돼도 프로토콜 수락이나 모델 실행 승인이 된 것은 아니다.

주요 로컬 산출물은 development_manifest.json, r1_inputs.jsonl, r1_gold.jsonl, calibration_manifest.json, calibration_tokens.json, preparation_report.json이다. development 모드에는 calibration 두 파일이 없다. 파일 내용·해시를 검토한 뒤 실제 ID·토큰 목록의 수락 여부를 판단한다. 문제·정답·원문 코퍼스는 공개 저장소에 올리지 않는다.

## 7. R0 실행기의 구체적인 범위

기본 명령은 네트워크나 모델 실행 없이 상태만 보여 준다.

~~~bash
python scripts/run_r0.py
~~~

현재는 기여 범위 판정·프로토콜 수락·연구 실행 승인이 미완료로 나온다. 이는 AGENTS.md 8번과 SESSION_PLAN.md에 기록된 조건이다. 해당 기록을 실제 근거와 함께 갱신하기 전에는 --execute를 사용해도 모델 로딩 전에 중단한다. 상태 문자열만 임의로 바꿔 조건을 충족시켜서는 안 된다.

승인 후 실행할 R0 후보는 다음과 같이 한정된다.

| 항목 | 실행 범위 |
|---|---|
| 모델 | 고정 revision의 DeepSeek-R1-Distill-Qwen-1.5B, BF16, Transformers 4.51.3/SDPA |
| 입력 | 기존 토크나이저 검사에서 사용한 합성 문제 2개 |
| 조건·시드 | BF16와 AWQ-W3, 각 입력 seed 42 |
| 출력 예산 | 응답당 128토큰, 총 4회 최대 512토큰. AWQ calibration 비용은 별도 |
| 기록 | 실제 입력·생성 ID, EOS 포함 토큰 수, 종료 사유, 시간·최대 할당 메모리, 계층 진단, 코드·자산 참조 |
| 실패 처리 | BF16 실행 오류 시 AWQ 탐색 중단. 중간 기록·계층 결과를 로컬 파일로 보존 |
| 성공 시 상태 | R0_EXECUTED_DIAGNOSTICS_PENDING_REVIEW |

~~~bash
# 관련 기록 수락 후의 실행 예시. 현재는 승인 조건에서 중단됩니다.
python scripts/run_r0.py --execute \
  --assets-dir results/local/runtime_assets_v03 \
  --upstream-dir results/local/awq_reference_v03 \
  --device cuda:0 \
  --output-dir results/local/r0_v03
~~~

종료 think 태그는 보존하고, EOS는 토큰 수에 포함한 뒤 디코딩용 끝 토큰에서만 제거한다. 이 실행기는 R1의 100문제 평가나 MATH-500 확증, KL 매칭, 장문맥·H1/H2 검정을 수행하지 않는다.

## 8. 남은 연구 판단

즉시 받을 다음 근거는 작은 무작위 Qwen2의 어댑터 점검 결과다. 그다음 실제 고정 데이터 파일의 출처 보고서와 선택 목록을 검토하고, 합성 R0 실행 조건을 수락해야 한다.

선행연구 충돌표를 통합한 최종 기여 판정과 연구 실행 승인 기록도 남아 있다. 기존 후보의 H1 수치 경계나 H2 방향은 이번 구현으로 동결하지 않았다. 세션 01을 유지하며, 세션 02 전환은 실제 종료 조건 충족을 확인한 뒤 알린다.

## 검토한 1차 자료

- [AWQ 원본 고정 commit](https://github.com/ruikangliu/Quantized-Reasoning-Models/tree/bf947e29f52e3f666e3263efac149dae0ac18d00): auto_scale.py, auto_clip.py, quantizer.py, module.py, qmodule.py, pre_quant.py와 MIT LICENSE.
- [Transformers v4.51.3 Qwen2](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py): 계층 입력·RoPE·attention 인자.
- [Hugging Face Hub v0.36.2 API](https://github.com/huggingface/huggingface_hub/blob/v0.36.2/src/huggingface_hub/hf_api.py): repo_info, RepoSibling, BlobLfsInfo.
- [GSM8K 파일 목록](https://huggingface.co/datasets/openai/gsm8k/tree/main/main), [MATH-500 파일 목록](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/main), [Pile-uncopyrighted 파일 목록](https://huggingface.co/datasets/monology/pile-uncopyrighted/tree/main): 경로·파일 형식 탐색. 실제 검증은 main이 아닌 고정 revision에 대해 수행한다.
