# QuantThink 재현 준비 v0.2 — 구현 범위와 실행 안내

작성일: 2026-09-29  
현재 단계: **QuantThink 01 — 저장소 점검·연구 설계 검증**  
판정: **CPU 준비 도구 32개 테스트 통과. 실모델 재현 준비 완료 판정은 보류.**

사용자가 제출한 토크나이저 검사 결과를 바탕으로, 다음 준비 작업인 AWQ 가중치 좌표 검사, GSM8K 정답 평가, 개발 문제 분리 도구를 구현했다. 이번 문서는 실행 승인이나 가설 검정 결과가 아니다. 기존 H1/H2와 R0/R1 후보 설정은 변경하지 않았다.

## 1. 이번에 완료한 작업

| 파일 | 구현한 기능 | 검증 범위 |
|---|---|---|
| [awq_weight_space.py](../scripts/awq_weight_space.py) | AWQ 스케일 기록의 순방향·역방향 적용, 원래 가중치 좌표에서 RMS 오차 계산 | NumPy 합성 계층의 수치 검사. 실모델 어댑터는 아직 없음 |
| [build_development_manifest.py](../scripts/build_development_manifest.py) | 문제 중복 제거, 확증 문제 제외, 해시 순서에 따른 개발 문제 선택 | 합성 JSONL 및 목록 검사. 실제 데이터셋 출처 인증·ID 선정은 아직 없음 |
| [evaluate_gsm8k.py](../scripts/evaluate_gsm8k.py) | 추론 이후 최종 영역의 수치 정답 평가, 문제·조건·시드별 누락 검사 | 합성 출력과 CLI 연동. 실제 생성문에 대한 평가기 감사는 아직 없음 |
| [test_reproduction_v02.py](../tests/test_reproduction_v02.py) | 좌표 복원, 정답 오인, 중복·누락·출력 보존 검사 | 새 CPU 테스트 32개 통과 |

원본 구현은 ruikangliu/Quantized-Reasoning-Models의 bf947e29f52e3f666e3263efac149dae0ac18d00을 기준으로 읽었다. 해당 코드나 모델을 실행하지 않았다. 이 저장소를 Lotfi 논문의 정확한 실행 환경으로 인증하지도 않았다.

## 2. 토크나이저 점검에서 유지할 조건

사용자 보고 상태는 TOKENIZER_CONTRACT_PASS_NOT_MODEL_READY이다. 자세한 근거는 [한글 검토 기록](TOKENIZER_CONTRACT_REVIEW_20260929.md)에 있다.

| 항목 | 앞으로 지켜야 할 조건 |
|---|---|
| BOS | 실제 토크나이저 BOS 151646을 사용한다. 모델 설정의 151643과 다른 사실을 기록한다. |
| 프롬프트 | 입력 템플릿의 시작 think 태그와 BOS를 다시 추가하지 않는다. |
| EOS와 패딩 | 둘 다 151643이므로 실제 attention mask를 보존한다. |
| 추론 종료 | 151649, 즉 종료 think 태그는 응답 전체의 EOS가 아니다. 최종 답변 영역을 유지한다. |
| 생성 설정 | T=0.6, top-p=0.95, top-k=0, repetition penalty=1.0을 명시한다. 원본 top-k=50을 그대로 상속하지 않는다. |
| 문맥 길이 | 토크나이저 16,384와 모델 설정 131,072는 메타데이터다. 후보 총 길이 40,960의 실동작은 미검증이다. |

아래 두 파일은 이번에도 수정하지 않았다. JSON 키 정렬·공백 제거·UTF-8 방식의 정규화 해시를 유지한다.

| 파일 | 정규화 SHA-256 |
|---|---|
| configs/reproduction_candidate.json | 98ff733b0d8391c3a33aa2f561e2d389bd6e482ee53b1b680c1bb6e7e6b6201f |
| configs/asset_inspection_refs.json | 97f5b7bf6cb6888c716a3838f21c224e7920b1df05c5996efe1321d39cd15276 |

설정 파일 안의 과거 미완료 표시를 현재 상태로 덮어쓰면 기존 보고서의 해시 연결이 끊어진다. 최신 증거는 [연구 상태](research_status.json)에 별도로 기록한다.

## 3. AWQ 가중치 비교의 기준 좌표

AWQ는 양자화 전에 인접 계층의 스케일을 바꿀 수 있다. 정규화 출력에 채널별 양수 스케일 s가 적용될 때 앞쪽 정규화 가중치를 s로 나누고, 다음 선형 계층의 해당 입력 열을 s배 하면 실수 연산에서 같은 함수를 나타낸다. 선형 계층 쌍도 앞쪽 출력 행과 bias를 s로 나누고 뒤쪽 입력 열을 s배 하는 형태다.

따라서 변환된 가중치와 원래 BF16 가중치를 바로 빼면, 기능을 보존하는 스케일 변화까지 양자화 오차로 섞인다. 이번 도구는 기록된 스케일 변환을 **역순으로 되돌린 뒤** 원래 참조 가중치와 비교한다. clipping과 rounding으로 생긴 손실은 역변환하지 않는다.

| 원본 Qwen2 경로 | 확인한 처리 | 이번 도구의 지원 범위 |
|---|---|---|
| input_layernorm → q/k/v | 정규화 스케일 분배 | norm_linears |
| post_attention_layernorm → gate/up | 정규화 스케일 분배 | norm_linears |
| up_proj → down_proj | up의 출력 채널과 down의 입력 채널 보정 | 전체 채널 수가 일치하는 linear_linear |
| v_proj → o_proj | 원본은 두 weight의 shape가 같을 때만 처리 | 실제 스케일 기록이 있고 차원이 맞는 경우만 지원. GQA에 강제로 적용하지 않음 |

스케일 기록 항목에는 kind, previous, following, scales가 필요하다. 누락된 파라미터, 비양수·비유한 스케일, 중복 계층, 지원하지 않는 변환은 실패로 처리한다. 입력 배열은 복사하므로 참조 값을 수정하지 않는다. 큰 실모델 전체를 NumPy float64로 복사하는 실행기를 뜻하지 않으며, 실제 어댑터에서는 계층 단위 처리와 dtype 정책이 필요하다.

CPU 테스트는 겹치는 두 스케일 변환이 있는 작은 gated MLP에서 다음을 확인했다.

- 양자화하지 않은 스케일 변환 전후 함수값이 float64 허용오차 안에서 일치한다.
- 역순 복원으로 원래 가중치를 회복한다.
- 변환 좌표에서 넣은 알려진 교란이 원래 좌표의 예상 크기로 남는다.
- 단순 변환을 양자화 오차로 오인하지 않는다.

여기서 사용한 1e-12 등의 수치는 합성 float64 테스트의 허용오차다. BF16 실모델의 logit 허용오차로 전용하지 않는다. 실제 AWQ 어댑터는 스케일·clip 기록을 내보내고, 양자화 전 스케일 변환과 양자화 후 좌표 복원 각각의 logit 차이를 별도로 검증해야 한다.

원본 W3 비대칭 양자화 경로는 그룹별 scale과 zero point로 반올림·clamp한 값을 다시 부동소수점으로 보관한다. 이를 통한 BF16 추론 시간은 native INT3 커널 속도 측정이 아니다. 이번 NumPy 도구 자체는 W3 양자화를 수행하지 않는다.

검토한 1차 코드:

- [auto_scale.py — 스케일 탐색과 적용](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/awq/auto_scale.py)
- [auto_clip.py — 그룹별 clipping 탐색](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/awq/auto_clip.py)
- [quantizer.py — 비대칭 가짜 양자화](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/awq/quantizer.py)
- [pre_quant.py — 계층 입력 포착 및 스케일·clip 적용](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/awq/pre_quant.py)

## 4. 런타임과 calibration 연결 조건

사용자가 보고한 환경은 Python 3.11.16, PyTorch 2.7.1+cu118, 토크나이저 점검용 Transformers 4.51.3이다. 원본 저장소의 requirements는 PyTorch 2.5.1, Transformers 4.47.1을 포함한다. 원본 requirements 전체로 현재 작동하는 환경을 교체하라는 지시가 아니다. 실제 어댑터의 의존성·버전은 호환성 검토 뒤 별도 고정한다.

후보는 두 조건 모두 같은 Transformers/SDPA 경로, worker당 GPU 1개, TP=1이다. [Transformers 4.51.3 Qwen2 코드](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py)에서는 상위 모델이 만든 position_embeddings가 attention에 전달된다. AWQ 탐색 중 블록을 직접 부를 때 attention mask, position 관련 인자와 포착한 kwargs를 보존해야 한다. 버전 문자열이 맞는 것만으로 블록 단위 실행 호환성을 인정하지 않는다.

Calibration 후보는 Pile-uncopyrighted의 고정 revision에서 **128 × 512 = 65,536 토큰**이다. 원본 [calib_data.py](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/awq/calib_data.py)는 문서 수를 센 뒤 연결해 블록을 만들므로 128문서와 128블록이 같지 않다. 원본의 pile-val-backup을 조용히 대체 사용하지 않는다.

실제 문서 ID·순서, source split, 문서 간 구분 토큰, 전처리, 토큰 해시는 아직 없다. 기존 exact_token_blocks 함수는 필요한 길이로 묶는 마지막 단계만 구현한다. 출처를 인증하는 calibration 로더와 AWQ 연결은 다음 구현 대상이다.

## 5. 개발 문제 선택과 데이터 분리

문제 선택 도구는 로컬 JSONL 파일 두 개를 입력으로 받는다.

| 입력 | 사용하는 내용 | 제약 |
|---|---|---|
| GSM8K main/train 후보 | question 문자열 | 공백 정규화 후 같은 문제를 중복 제거 |
| MATH-500 확증 문제 목록 | problem 문자열 | 정확히 500행·500개 고유 문제 해시 필요. 정답 필드는 선택에 사용하지 않음 |

공백 정규화는 연속 공백을 한 칸으로 바꾸는 처리다. 대소문자·수식·Unicode를 추가로 통합하지 않는다. 개발 문제 중 확증 문제와 같은 해시를 가진 항목을 제외하고, SHA256(20260929 + 줄바꿈 + 정규화 문제) 순서로 처음 100개를 고른다. ID는 gsm8k/train/와 문제 해시를 연결한다.

입력 행의 순서를 바꿔도 선택 결과는 같다. 부족한 문제를 재표집하거나 다른 split에서 채우지 않는다. 출력에는 선택 ID, 문제 해시, 제외·중복 개수, 입력 파일 바이트 해시가 남으며 문제 본문과 정답은 남기지 않는다. 생성 결과는 기존 파일을 덮어쓰지 않는다.

**로컬 파일 해시는 원격 출처 인증이 아니다.** 도구는 명시된 revision의 공식 파일을 다운로드했는지 증명하지 않는다. 부분적인 GSM8K 파일도 충분한 행 수만 있으면 후보를 만들 수 있으므로, 실제 실행용 목록으로 수락하기 전에 원본 경로·split·revision·전체 행 수를 확인해야 한다. 이 때문에 출력 상태는 LOCAL_CANDIDATE_NOT_SOURCE_AUTHENTICATED이다.

MATH-500은 질문 해시 중복 제외에만 사용한다. 정답을 이용한 선택, 생성, 튜닝은 하지 않는다. 질문 해시 검사는 의미 중복이나 사전학습 오염 부재를 보장하지 않는다. 남은 GSM8K 문제는 KL 매칭·매칭 검증에 아직 배정하지 않았으며, R1 개발 문제를 그 용도로 재사용하지 않는다.

## 6. GSM8K 평가 규칙과 한계

입력 generated_text는 **프롬프트를 제외한 생성 영역**이다. 입력에 미리 들어 있던 시작 think 태그를 붙이지 않는다. 디코딩 전에 끝의 EOS와 배치 패딩을 토큰 ID로 구분해 처리하고, 종료 think 태그는 텍스트에 보존해야 한다. 생성 토큰 수는 EOS를 포함해 별도 기록한다.

| 상황 | 처리 |
|---|---|
| 추론 부분에만 정답 숫자가 있음 | 최종 정답으로 인정하지 않음 |
| 종료 think 태그가 없거나 두 개 이상 있음 | 명확한 최종 영역 없음 |
| 생성 영역에 시작 think 태그가 다시 나타남 | 경계 위반으로 별도 기록 |
| 최종 영역의 boxed 수치 정답 | 중괄호를 맞춰 읽고 정수·소수·분수를 정확한 유리수로 비교 |
| 명시적인 Answer: 또는 The final answer is 행 | 지원 문법의 숫자인 경우 비교 |
| 여러 boxed/정답 행의 숫자가 서로 다름 | 모호한 최종 정답 |
| 동일 수치의 소수·분수 표현이 반복됨 | 같은 값으로 인정 |
| 일반 문장에 마지막 숫자만 있음 | 임의로 추출하지 않음 |
| 단위·백분율·기호식·지원하지 않는 LaTeX | 자동 제거하거나 실행하지 않고 파싱 실패로 기록 |

이 문법은 일부 표현을 보수적으로 거부한다. boxed 바깥의 모든 자연어 모순을 이해하는 평가기가 아니므로, 파싱 성공·실패 모두 실제 생성문 표본으로 감사해야 한다. MATH-500 기호식 평가기는 이번 범위에 없다.

정답 JSONL에는 id와 answer가 필요하며, answer는 원본 GSM8K의 #### 구분자를 포함한다. 생성 JSONL에는 id, arm, seed, finish_reason, generated_text가 필요하다. id는 선택 목록의 ID를 사용하고 arm은 bf16 또는 awq_w3, seed는 42 또는 43이다. 정상 종료 사유는 eos 또는 length이며 실행 실패는 error 또는 unexpected_stop으로 남긴다.

CLI는 개발 목록의 설정 해시·ID 수·선택 해시를 검사하고, 정답 ID가 계획된 목록 전체와 정확히 일치해야 진행한다. 임의로 정답 파일을 줄여 평가 분모를 축소할 수 없다. 목록 내용과 원본 정답의 진위는 별도 출처 검토 대상이다.

| 집계 항목 | 처리 |
|---|---|
| 계획된 문제 × 조건 × 시드 | 전부 출력에 유지 |
| 토큰 상한 도달 | 평가 분모에 포함. 최종 정답이 없으면 파싱 실패도 기록 |
| 파싱 실패 | 후보 자동 점수에서는 오답 처리하고 parse_failures로 별도 집계 |
| 누락·OOM 등 실행 실패 | 정오를 null로 유지. 해당 조건의 primary_accuracy도 null |
| 완료분 정확도 | completed_accuracy_descriptive로만 표시 |
| 전체 평가가 불완전함 | INCOMPLETE_GRID_NO_PRIMARY_COMPARISON, CLI 종료 코드 2 |
| 전체 평가가 완전함 | COMPLETE_GRID_CANDIDATE_SCORING. 평가기 승인이나 H1 검증 통과를 뜻하지 않음 |

파싱 실패가 있으면 requires_parse_audit=true이며, main_hypothesis_ready는 항상 false이다. 조건마다 답안 형식이 다르면 파싱 실패율 자체가 정확도 차이처럼 보일 수 있다. 사람의 감사와 규칙 검토가 끝나기 전에는 이를 능력 저하나 과잉 추론 증거로 해석하지 않는다. 실제 데이터 출처도 자동 인증하지 않으므로 requires_source_provenance_review=true를 남긴다.

길이 집계·문제 단위 bootstrap·가설 검정은 이 평가기에 구현하지 않았다.

## 7. 정답 포기 현상의 수동 판정 초안

숫자가 추론에 한 번 등장했다는 사실만으로 올바른 중간 풀이를 확인했다고 보지 않는다.

| 후보 분류 | 필요한 확인 |
|---|---|
| VALID_SOLUTION_THEN_WRONG_FINAL | 올바른 풀이가 실제로 완성된 뒤, 이를 버리고 잘못된 최종 답을 냄 |
| VALID_SOLUTION_THEN_NO_FINAL | 올바른 풀이가 실제로 완성됐으나 최종 답을 확정하지 않음. 상한 종료 여부를 함께 기록 |
| INCIDENTAL_ANSWER_STRING | 정답 숫자가 우연히 등장했으며 올바른 풀이 완성은 확인되지 않음 |
| OTHER_ERROR | 위 분류에 속하지 않는 계산·추론 오류 |
| UNCERTAIN | 타당성 또는 포기 여부를 확정하기 어려움 |

판정자는 조건 이름을 가린 출력으로 평가하고, 자동 후보의 양성뿐 아니라 음성 표본도 확인한다. 여러 현상이 동시에 보이면 위 표의 첫 번째 분류로 임의 처리하지 않고 근거를 남긴다. 표본 수·판정자 수·합의 절차·일치도 기준은 아직 수락되지 않은 설계 항목이다. 이 표만으로 overthinking 지표의 타당성이 확보되지는 않는다.

## 8. 지금 사용자가 실행할 명령

기존 quantthink 환경에서 변경을 받은 뒤 CPU 테스트만 실행한다. 토크나이저 점검은 다시 할 필요가 없다.

~~~bash
cd ~/quantthink
git switch setup/session-01-research-gates
git pull --ff-only
python -m unittest discover -s tests -p 'test_reproduction_v02.py' -v
~~~

새 테스트는 표준 라이브러리와 NumPy를 사용한다. 모델·데이터셋 다운로드, CUDA 호출, 패키지 설치는 하지 않는다. 기대 출력은 Ran 32 tests와 OK이다.

이 문서 작성 환경에서는 Python 3.12.14, NumPy 2.3.5로 32개가 통과했다. 사용자 환경인 Python 3.11.16, NumPy 1.26.4에서의 실행 결과는 아직 제출되지 않았다. 이전 테스트 묶음과 토크나이저 검사를 이번 결과에 합산하지 않았다.

다음 명령은 나중에 출처를 확인한 로컬 데이터와 생성 결과가 준비됐을 때 사용할 인터페이스 예시다. 지금 해당 파일이 존재하거나 실험이 승인됐다는 뜻은 아니다.

~~~bash
python scripts/build_development_manifest.py \
  --development results/local/gsm8k_train.jsonl \
  --confirmation results/local/math500_questions.jsonl \
  --output results/local/development_manifest_v02.json

python scripts/evaluate_gsm8k.py \
  --development-manifest results/local/development_manifest_v02.json \
  --gold results/local/r1_gold.jsonl \
  --generations results/local/r1_generations.jsonl \
  --output results/local/r1_scores_v02.json
~~~

개발 정답 파일과 생성 결과를 만드는 인증된 데이터 준비·모델 실행기는 아직 없다. 공개 저장소에는 원본 문제·정답·대형 모델·로컬 진단 원문을 올리지 않는다.

## 9. 다음 구현과 세션 전환 조건

다음 구현 대상은 고정 revision에서 calibration과 개발 데이터를 인증해 읽는 로더, 실제 Qwen2용 AWQ 어댑터, 생성 토큰·EOS·시간·메모리를 남기는 공통 실행기다. 이후 실제 출력에 대한 평가기 감사와 수동 판정 검증, R0 로더·종료 검사, 장문맥 가능성 검증이 필요하다.

연구 설계에서는 선행연구 충돌표를 통합한 기여 판정, 재현 프로토콜 수락, 연구 실행 승인 기록이 남아 있다. AGENTS.md의 연구 실행 조건에 따라 현재 세션은 01로 유지한다. 허용된 문헌 검토·설계·CPU 구현 작업은 이어갈 수 있으며, BF16/AWQ 모델 실험·KL 매칭·H1/H2는 아직 실행하지 않았다.

