# 세션 01 → 02 인계 — R0 실행 준비

**현재 세션 이름: QuantThink 02 — 환경 구축·BF16/AWQ 재현**

세션 01의 종료 조건을 충족했다. 2026-09-29 15:34:45 한국시간의 사용자 승인으로 제한된 기여 범위·고정 calibration 후보·R0 계획의 수락 및 실행 승인을 기록했다. [승인 기록](R0_APPROVAL_20260929_KO.md)을 참조한다.

## 인계 상태

| 항목 | 확인한 상태 |
|---|---|
| 저장소 | JunKIM2603a/quantthink |
| 작업 브랜치 | setup/session-01-research-gates. 세션 이름이 바뀌어도 기존 브랜치를 계속 사용 |
| 준비 기준 commit | ff14820c6fde1f4b4b907ceacc6b65ad9bb2cb53. 실행 전 이 승인 기록이 추가된 최신 브랜치를 pull |
| 검토 PR·이슈 | [Draft PR #1](https://github.com/JunKIM2603a/quantthink/pull/1), [세션 01 Issue #2](https://github.com/JunKIM2603a/quantthink/issues/2) |
| 완료한 설계 근거 | 선행연구 관련 본문 검토·통합 충돌표·제한된 기여 범위 수락 |
| 사용자 실행 근거 | CUDA·토크나이저·작은 무작위 Qwen2 AWQ 점검·전체 데이터 준비 보고 |
| 독립 대조 근거 | 원본 데이터 파일 3개 해시, 개발 산출물 3개 재구성 해시 일치 |
| Calibration 근거 | 사용자 보고 261문서·128 × 512토큰. 파일·선택·토큰 해시를 고정했으며 실제 로컬 바이트는 R0 실행 전에 대조 |
| 확정 프로토콜 | 기존 v0.3 구현과 R0 검토 문서의 합성 2문제 × BF16/AWQ·응답당 128토큰 범위 수락 |
| 가설·후속 프로토콜 | H1/H2 수치 기준·본검정 및 R1은 DRAFT |
| 실모델 결과 | R0·BF16/AWQ 재현·KL 매칭·H1/H2 모두 NOT_RUN |

준비·토크나이저·합성 GPU 점검을 반복하지 않는다. 다음에 할 단 하나의 작업은 **기존 사용자 환경에서 승인된 R0를 실행해 첫 실모델 진단을 확보하는 것**이다.

## 실행 명령

~~~bash
conda activate quantthink
cd ~/quantthink
git switch setup/session-01-research-gates
git pull --ff-only

python scripts/run_r0.py --execute \
  --assets-dir results/local/runtime_assets_v03 \
  --upstream-dir results/local/awq_reference_v03 \
  --device cuda:0 \
  --output-dir results/local/r0_v03
~~~

기존 환경의 torch 2.7.1+cu118, Transformers 4.51.3, Hub 0.36.2를 사용한다. 실행기는 승인·추적 파일의 Git 상태·런타임·자료 해시를 검사한 후 고정 모델을 로딩한다. 합성 문제별 seed 42, T=0.6, top_p=0.95, top_k=0, repetition_penalty=1이다. 총 4회 생성·최대 512토큰이며, 전체 계층 AWQ calibration 탐색 시간은 별도다.

산출물은 `results/local/r0_v03/run.json`과 `awq_layer_*.json`이다. 실행 완료 상태는 `R0_EXECUTED_DIAGNOSTICS_PENDING_REVIEW`이며 model_ready는 false다. 정확도·동등성·장문맥의 인증이 아닌 실행 및 진단 검토 대기 상태다. 오류가 나면 터미널 오류와 남은 run.json을 보존한다. 이미 생성된 출력 폴더를 지우거나 덮어쓰지 않는다.

## 실행 후 공유할 요약

아래 명령은 산출물을 읽기만 한다. 생성 ID 전체와 calibration 토큰 배열은 출력하지 않는다. 합성 문제의 생성 텍스트와 종료·성능·계층 차이 요약을 모은다.

~~~bash
conda activate quantthink
cd ~/quantthink
python - <<'PY'
import json
from pathlib import Path

p = Path("results/local/r0_v03/run.json")
r = json.loads(p.read_text())
keys = (
    "status", "recorded_at_utc", "repository", "packages", "python",
    "candidate_sha256", "policy_sha256", "preparation_report_sha256",
    "prepared_assets_review_sha256", "research_status_sha256",
    "completed_awq_layers", "awq_status", "model_ready", "error_type", "error",
)
summary = {k: r[k] for k in keys if k in r}
summary["attempts"] = [
    {k: row[k] for k in (
        "fixture_id", "arm", "seed", "input_token_count",
        "generated_token_count_including_eos", "finish_reason", "budget_reached",
        "right_censored", "generated_text", "elapsed_seconds", "peak_allocated_bytes",
        "error_type", "error",
    ) if k in row}
    for row in r.get("attempts", [])
]
summary["layer_diagnostics"] = r.get("awq_layer_diagnostics", [])
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
~~~

실행 중 실패하면 완료된 계층 JSON이 별도로 남을 수 있다. 요약에 계층 차이가 없다고 앞선 계층 작업이 없었다고 해석하지 않는다. 실패 위치와 남은 파일을 함께 검토한다.

## 남은 검토와 다음 세션 조건

첫 R0에서 로딩·변환·생성·종료·계층 진단을 검토한다. EOS 종료와 예산 도달을 구분하고, 소형 모델에서 사용한 상대 RMS 5% 한도를 실제 모델 동등성 인증에 적용하지 않는다. 후속 실모델 좌표·logit 검증과 생성문 평가·수동 판정, R1 재현 기준·장문 예산은 별도 계획으로 확정한다.

현재 승인으로 R1 100문제·MATH-500·KL 매칭·장문맥·H1/H2·7B를 실행하지 않는다. 세션 02 종료 조건을 충족한 뒤 **QuantThink 03 — 교란 매칭·대조군 검증** 전환을 판단한다.

## 새 채팅에서 이어갈 때

~~~text
세션 이름: QuantThink 02 — 환경 구축·BF16/AWQ 재현
저장소: https://github.com/JunKIM2603a/quantthink
브랜치: setup/session-01-research-gates

저장소의 AGENTS.md, docs/research_status.json, docs/SESSION_PLAN.md,
docs/R0_APPROVAL_20260929_KO.md, docs/HANDOFF_01_TO_02_KO.md를 먼저 읽고
실제 최신 브랜치·커밋·보고서를 확인해 이어서 진행해줘.

세션 01은 완료됐다. 제한된 기여 범위와 고정 자료를 사용하는 R0 계획은
2026-09-29의 명시적 사용자 승인으로 수락·실행 승인 기록을 마쳤다.
R0는 합성 2문제 × BF16/AWQ-W3, 응답당 최대 128토큰, cuda:0 범위다.
같은 범위의 승인을 다시 요청하지 마라. 실행 결과는 실제 보고서로 확인해야 한다.

다음 작업은 승인된 R0 실행·결과 검토다. 기존 데이터 준비·토크나이저·합성 AWQ
점검을 반복하지 마라. H1/H2·R1·확증·KL 매칭·장문맥·7B까지 승인된 것은 아니다.
문서는 한글로 작성하고, 명령은 conda activate quantthink과 cd ~/quantthink으로 시작해줘.
이 인계문만으로 모델 실행 완료나 가설 지지를 추정하지 마라.
~~~
