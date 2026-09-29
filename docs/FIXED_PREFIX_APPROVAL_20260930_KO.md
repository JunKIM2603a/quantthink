# 고정 prefix 진단 v01 실행 승인 — 2026-09-30

현재 세션: **QuantThink 02 — 환경 구축·BF16/AWQ 재현**

2026-09-30 00:41:13 한국시간에 사용자가 직전 최대12회 진단 제안에 대해 **“다음 연구 단계 진행해줘.”**라고 지시했다. 이를 아래에 이미 제시한 구체적 범위의 수락과 실행 승인으로 기록한다. 사용자 발화에 숫자별 승인이 따로 있었다고 바꾸어 기록하지 않는다. 기존 R0 승인 재요청은 없고 이 범위의 승인을 다시 묻지 않는다.

## 승인 대상

- 제안과 실행 코드 기준 commit: `8785736bce854c54e2644fab55b5f4f3ccc6e217`.
- 프로토콜: [고정 prefix 진단 v01](FIXED_PREFIX_DIAGNOSTIC_V01_KO.md).
- 설정: `configs/fixed_prefix_diagnostic_v01.json`. 정규화 SHA-256 `5d24000bfa9d7b15d398b978f671d94be816918bd9c5881a458448bced488fd6`.
- D2-03/D3-01의 기존 BF16 seed42 응답에서 입력과 생성 앞2048토큰을 고정 재사용한다. 전체 문맥2119/2127, 입력별129개 위치의 전체 vocabulary logit을 측정한다.
- B/B_repeat/S/Q/C/Cw, 최대12회 full-sequence forward, 처리입력 합계25476토큰, 새 생성0토큰. BF16 반복이 비동일하면4회에서 중단한다.
- 기존 cuda:0·1.5B·BF16·SDPA·고정 패키지를 사용한다. 진단용 config 복사본만 sliding_window=None으로 두고 use_cache=false로 실행한다. 저장된 AWQ recipe 재적용과 좌표 복원을 포함하며 새 calibration 탐색은 없다.
- 원시 logit 데이터 본체 최대940787712바이트와 JSON/헤더 공간을 사용한다. 원시 배열은 로컬에 보존한다. 모델 로딩·파라미터 해시·변환 비용은 forward 횟수와 별도다.

승인 범위의 설정·코드·예산은 앞 제안과 동일하다. 아래 실행 코드 해시를 `research_status.json`의 승인에 연결했다.

| 파일 | SHA-256 |
|---|---|
| `run_fixed_prefix_diagnostic.py` | `8c0c5d72984c8ba366e2209657f2bacbede196cbb52c1f341a08d4f942f326ff` |
| `fixed_prefix_diagnostic_contracts.py` | `bafc0997feac28d93a2fc5b30910fe487edea85ccb56f9adfeaf4a14fc00c31d` |
| `replay_r0_awq.py` | `b01e0124eb97a7e1145c886842d894966084ac8b1f7b47f538c647d7e1fb574a` |
| `qwen2_awq_adapter.py` | `ea7d9e3c35443fc889480c667a826a9eca793c7592fb03be778b150d094b59c5` |
| `awq_weight_space.py` | `230b604c0f436696b9118753c8ddbbd383029a3ca9a2aa89aab6478b1bd23b6b` |
| `runtime_assets.py` | `d647bb31a0a72a373a95bf1561603c8eff2563ff598eff60516bd54dbedca130` |
| `run_difficulty_pilot.py` | `809b14475e7eac56ce6667b52b85839e674f766106130407b6802aafe9fdaec7` |
| `difficulty_pilot_contracts.py` | `d14b79ea22069dc6666fce2e70edc17557ef97029158eafdd51ebb306c460446` |
| `runtime_contracts.py` | `ca514a25f560b8912edd90de83cb8d4a18aed0cd97d8769f5b9f1627cc8924ee` |
| `reproduction_contracts.py` | `04923d60075be31ca7b14b7566dd0040b474bb4f1e009d416ee66819401e45bb` |

## 승인 당시 실행 명령 이력 — 재실행하지 않음

아래는 승인 당시 제공한 명령의 이력이다. 이후 사용자12회 완료 보고를 받아 [결과 검토](FIXED_PREFIX_RESULT_REVIEW_20260930_KO.md)를 마쳤다. 현재 명령은 해당 검토 문서의 CPU 원시 배열 감사이며 아래 GPU 명령을 다시 실행하지 않는다. 이 대화 환경에서 사용자 GPU 실행을 시작하거나 직접 관찰한 것은 아니다.

```bash
conda activate quantthink
cd ~/quantthink
git switch setup/session-01-research-gates &&
git pull --ff-only &&
python scripts/run_fixed_prefix_diagnostic.py --execute
```

검토할 파일: `results/local/fixed_prefix_v01/run.json`. `logits_*.npy`는 로컬에서 보존하고 공개 Git에 올리지 않는다. 출력 폴더가 이미 있거나 오류/중단이 발생하면 결과를 지우거나 자동 재실행하지 않고 현재 파일과 로그를 검토한다.

## 승인과 결과의 구분

승인 기록 작성 당시 상태는 `APPROVED_AWAITING_USER_GPU_EXECUTION`이었다. 당시 코드/설정과 순수 Python 승인 조건을 대조했으며 GPU 실행은 하지 않았다. 이후 실행 commit907ca61의 12회 완료 run.json을 받았고 승인 해시와 일치함을 확인했다. 현재 상태는 `REPORT_REVIEWED_SAVED_LOGIT_AUDIT_PENDING`이며 승인 예산12회는 소진됐다. 원시 `.npy`와 가중치 배열의 독립 검증 또는 기능적 동등성 인증과 구분한다. 준비 시점 CPU 검사7개 기록은 그대로 유지한다.

수집 완료는 기능적 동등성 PASS가 아니다. `model_ready=false`, `next_session_ready=false`, 세션02를 유지한다. 기존 R0/20문제 결과·자동 점수·H1/H2는 변경하지 않는다. R1·MATH-500·새 Pile 추론·KL 매칭·4096 경계 영향 실험·7B·추가 생성이나 자동 예산 확대는 이번 승인에 포함하지 않는다.
