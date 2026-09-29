# 설치 코드의 SDPA 마스크 확인 결과 — 세션 02

**CPU 마스크 확인은 완료됐다. `use_sliding_window=false`여도 기록 설정의 `sliding_window=4096`이 이전 키를 차단하는 동작을 관측했다.** 모델 forward·SDPA 연산·GPU·원래 생성에 미친 영향의 검증은 아니다. 이 파일을 다시 만들도록 요청하지 않는다.

## 근거와 검증 깊이

- 검토한 브랜치 HEAD: `c45899c7d02056422e5aaa9d58a8bb183a50f767`.
- 첨부: `sdpa_mask_review_v01.json`, 4,556바이트, SHA-256 `bd18f808619c03859653d43c6ad956f3aea7c9d816a27e8387ff0b7fe3b81b0c`.
- 보고 시각: 2026-09-29 14:55:42.644433 UTC. 실행 commit 필드는 없으며, 보고된 검사 스크립트 SHA-256 `831ce0cc8ad55cd68f55f280941a28cf7408a0e4f4e13fcc7829e707ee9928b3`이 위 HEAD의 파일과 일치한다.
- 원본 근거 묶음 SHA-256 `f9e30e7ad4b7638689e62a4f9f3368d61b5ac21d735a60cb08054019757baad4`, 모델 설정 정규화 SHA-256 `3aa3cb40c0c8ecafe298c52184816832c6602b9cd4e8c75cb9f9405a64480ee2`가 기존 검토와 일치한다.
- torch 2.7.1+cu118 / Transformers 4.51.3, CPU BF16, query 길이 1, DynamicCache 조건이다.

첨부의 8개 행에서 위치·KV 길이·분기·차단 개수와 처음/마지막 키를 재계산했다. 설치 파일로 보고된 아래 4개 소스는 공식 `huggingface/transformers@v4.51.3`에서 직접 받은 바이트와 Git blob SHA-1 및 SHA-256이 모두 일치한다. 사용자 CPU 과정을 원격 관찰한 것은 아니며, 작성 환경에는 PyTorch/Transformers가 없어 텐서 실행을 독립 반복하지 않았다.

| 소스 | Git blob SHA-1 |
|---|---|
| configuration_qwen2.py | `2e82f1976f3922f3620415f4eace6c6e046243f8` |
| modeling_qwen2.py | `16a7316e2d0e56eafe301a7f2d8693d6cc6c73ec` |
| modeling_attn_mask_utils.py | `dfdd976f0156139024c6f7788870a71e4a41754d` |
| sdpa_attention.py | `9c924c048ad52929a2d0f890a22295d5a56ef505` |

원본: [설정](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/configuration_qwen2.py), [모델](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py), [마스크](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/modeling_attn_mask_utils.py), [SDPA 연결](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/integrations/sdpa_attention.py).

## 관측과 의미

| query 위치(0부터) | KV 길이 | 기록 설정의 마스크 | 제외된 이전 키 | 메모리에서 window=None인 대조의 제외 키 |
|---:|---:|---|---|---:|
|4094|4095|명시적 마스크 없음|0개|0개|
|4095|4096|명시적 4차원 마스크|0개|0개|
|4096|4097|명시적 4차원 마스크|0번, 1개|0개|
|4204|4205|명시적 4차원 마스크|0~108번, 109개|0개|

대조 조건도 `use_sliding_window=false`다. 차이는 메모리에 만든 config의 `sliding_window` 하나다. 원본 설정·패키지·생성 결과 파일은 수정하지 않았다고 보고됐고, 검사 코드에도 해당 쓰기 경로가 없다. query 길이 1의 implicit SDPA 경로에서는 모든 기존 키가 과거/현재 위치다. 보고서의 `implicit_sdpa_causal`은 마스크 미반환 분기 이름이며, 실제 SDPA kernel을 호출해 확인했다는 뜻이 아니다.

고정 소스의 공통 마스크 함수가 window 값을 직접 읽는 동작과 일치한다. 경고가 Ctrl+C를 일으킨 것은 아니지만 이번 긴 응답에 무관한 경고라고 취급해서는 안 된다. `DISABLED_FLAG_WINDOW_MASK_OBSERVED`는 이 조건의 마스크 관측 결과이며 모델 동등성 PASS/FAIL이 아니다.

## 기존 20문제 결과에 대한 판정

기존 [원본 검토](DIFFICULTY_EVIDENCE_REVIEW_20260929_KO.md)의 점수·생성문·4096 출력 상한은 그대로 보존한다. 전체 입력+생성이 4096을 넘는 기록은 44개(BF16 16, AWQ 28)다. 당시 cache 객체·실제 마스크를 저장하지 않아 이 44개 모두에서 동일한 영향을 직접 확인한 것은 아니다.

공통 생성 prefix 3986토큰까지만 재검토한 기존 사후 집계에서도 EOS는 BF16 24/40, AWQ 12/40이다. 마지막 예측의 KV 길이는 최대4095로 이번 분기 이전이다. 따라서 전체 종료 차이를 경계 마스크 하나로 설명할 수 없다. 이 관측만으로 양자화의 순수 인과 효과·과잉 추론·정답 포기·전체 자연 길이 증가를 확정하지도 않는다.

수동 답안 보완에서 BF16 D1은 8/8 EOS·정답 조건을 충족했고 AWQ는 D1부터 검열이 있다. 원래 v01 자동 난도 판정은 모두 INCONCLUSIVE다. 이번 CPU 결과가 이를 소급 변경하지 않는다.

## 다음 작업

[고정 prefix 진단 v01](FIXED_PREFIX_DIAGNOSTIC_V01_KO.md)에 실제 D2·D3 응답의 고정 prefix 2개, B/S/Q/C/Cw와 BF16 반복, 최대12회 full-sequence forward를 준비했다. 새 생성 0토큰이며 저장된 recipe만 재적용한다. 새 진단에 한해 full attention을 명시하려고 원본 설정의 복사본에서 `sliding_window=None`을 적용하고 원본/유효 설정 해시를 각각 기록한다. 기존 난도 실행기나 결과를 바꾸지 않는다.

이 GPU 계산은 새 범위이며 아직 실행하지 않았다. 저장소 AGENTS.md 8번의 연구 실행 gate와 사용자의 자동 예산 확대 금지에 따라 범위를 검토한 뒤 실행한다. 기존 R0나 완료한 80개 시험의 재승인을 요구하는 것이 아니다. `--plan`은 모델·토크나이저·GPU 없이 열람할 수 있다. 추가 CPU 마스크 반복·의존성 업데이트·flash-attn 설치·20문제 재시험은 다음 명령에 포함하지 않는다.

기능적 좌표/양자화 검증, 생성 cache 및 문맥 경계의 실제 영향, 일반 평가 타당도와 재현 기준은 남아 있다. **세션02, model_ready=false, next_session_ready=false를 유지한다.**
