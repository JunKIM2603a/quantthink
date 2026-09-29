# 종료 prefix 진단 v01 실행 승인 — 2026-09-30

현재 세션: **QuantThink 02 — 환경 구축·BF16/AWQ 재현**

2026-09-30 06:27:01 한국시간에 사용자가 직전 추가 최대6회 진단 제안에 대해 **“진행해줘”**라고 지시했다. 이를 이미 제시한 아래 구체적 범위의 수락과 실행 승인으로 기록한다. 사용자 발화에 숫자별 승인이 따로 있었다고 기록하지 않는다. 같은 범위의 승인을 다시 요청하지 않는다.

## 승인 대상

- 제안과 실행 코드 기준 commit: `f8fb2ee7fb084c3534f52cf20bca458ed104aa0d`.
- 프로토콜: [종료 직전·AWQ 경로의 고정 prefix 진단 v01](TERMINATION_PREFIX_V01_KO.md).
- 설정: `configs/termination_prefix_v01.json`. 정규화 SHA-256 `073986759aa41f29e54dfdf2835e7877aa2a2dccdb7347878a696451bdc2b1b3`.
- 기존 D2-03 seed42의 BF16 종료 직전 경로와 같은 길이 AWQ 경로, 입력2개를 저장 ID에서 고정 재사용한다. 각각 prompt71+생성2424=2495토큰이며 EOS는 입력에 포함하지 않는다. 각 입력의2366~2494번 위치129개에서 전체 vocabulary logit을 측정한다.
- B/B_repeat/Q, **추가 최대6회 full-sequence forward·처리입력14970토큰·새 생성0토큰**. BF16 반복이 비동일하면4회에서 중단하고 Q는 실행하지 않는다. 자동 재시도·입력 추가·예산 확대를 포함하지 않는다.
- 기존1.5B revision·cuda:0·BF16·SDPA·고정 패키지를 사용한다. 진단용 설정 복사본은 `sliding_window=None`, `use_cache=false`다. 기존28계층 AWQ scale/clip/W3 g128 recipe 재적용을 포함하며 calibration 탐색은 하지 않는다.
- 원시 logit6개 `[129,151936]` float32의 본체470393856바이트(약449MiB)와 헤더/JSON 공간을 사용한다. 모델 로딩·해시·recipe 재적용 비용은 forward 횟수와 별도다.
- T=1 전체 어휘 비교와 T=0.6/top-p=.95 float64 수학적 필터를 기록한다. GPU FP32 샘플링·RNG·cache 경로 재현 또는 새 자유 생성이 아니다.

설정·코드·입력·예산은 제안과 같다. 연구 상태에 `authorization=APPROVED`, `remaining_forward_calls=6`과 아래12개 코드 해시를 연결했다. 설정 JSON의 `PROPOSED_NOT_AUTHORIZED_NOT_RUN` 및 기존 결과 검토 JSON의 승인 대기 문자열은 고정된 당시 스냅샷이다. 현재 승인은 이 문서와 `research_status.json`의 `termination_prefix_diagnostic`를 따른다.

| 파일 | SHA-256 |
|---|---|
| `run_termination_prefix_diagnostic.py` | `a65f5e0484e03fc64eef698f81d24c1b3db6a9e1012e2c1d1f9f6216b44b270e` |
| `audit_fixed_prefix_logits.py` | `ab9b164c32413cdc4a2d8015c531b7a15bac0b76ebe7a5b7d9bc14894a2f5abd` |
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

## 실행 명령과 제출할 결과

사용자 GPU 환경에서 다음 명령을 1회 실행한다. 기존20문제·고정 prefix12회·CPU 감사는 완료됐으며 반복하지 않는다. 입력 계획도 실제 첨부로 대조했으므로 `--plan` 출력 재제출은 필요 없다.

```bash
conda activate quantthink
cd ~/quantthink
git switch setup/session-01-research-gates &&
git pull --ff-only &&
python scripts/run_termination_prefix_diagnostic.py --execute
```

검토할 파일은 **`results/local/termination_prefix_v01/run.json`**이다. 원시 `.npy`는 로컬에 보존하고 공개 Git에 올리지 않는다. 정상 수집 상태는 `TERMINATION_PREFIX_COLLECTED_PENDING_REVIEW`다. B 반복이 다르면 `TERMINATION_PREFIX_BASELINE_REPEAT_DIFFERENCE`로 중단한다. 출력 폴더가 이미 있거나 오류/중단이 발생하면 결과를 지우거나 자동 재실행하지 않고 현재 파일과 로그를 검토한다.

## 승인과 결과의 구분

현재 상태는 `APPROVED_AWAITING_USER_GPU_EXECUTION`이다. 사용자 실행 결과는 아직 받지 않았고 이 대화 환경에서 GPU/모델을 실행하지 않았다. 이번 갱신에서는 최신 브랜치와 제안 설정·코드12개 해시가 같은지 및 순수 Python 승인 조건만 확인했다. 준비 당시 통과한 새 CPU 검사6개와 실제 첨부 입력 대조 기록은 보존하며 다시 실행하지 않았다.

한 문제·한 시드의 사후 선택 탐색이다. BF16 경로 마지막 위치에서 B의 EOS 후보 포함 여부를 먼저 확인하고, 두 경로 각각의 B/Q 종료 분포를 비교한다. 자연 길이 지연 원인 확증·기능적 동등성 PASS·H1/H2 검정으로 해석하지 않는다. 자세한 분류 규칙은 프로토콜을 따른다.

`model_ready=false`, `next_session_ready=false`, 세션02를 유지한다. 기능적 수락 한도·독립 확인·평가기 일반 타당도·생성/cache 영향과 종료 원인 해석은 남아 있다. 기존 R0/20문제 결과·자동 점수·H1/H2는 변경하지 않는다. R1·MATH-500·새 Pile 추론·KL 매칭·장문맥·7B·추가 생성·자동 예산 확대는 이번 승인에 포함되지 않는다.
