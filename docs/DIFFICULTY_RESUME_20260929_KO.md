# 난도 시험 중단 로그 검토와 재개 — 2026-09-29

현재 세션: **QuantThink 02 — 환경 구축·BF16/AWQ 재현**

## 확인한 상태

사용자가 첨부한 터미널 로그를 직접 읽었다. 첨부 SHA-256은 `52907e27e8918e6e55fcde1d79392be872f87853faa4df19611caee27fe936b2`다. 로그의 실행 코드는 fbe673e97a42e38942745aea6d028a3f58bf0b90으로 갱신된 상태다. 사용자 원본 run.json·전체 생성 ID·정답 텍스트는 받지 않았으므로 아래는 **로그 보고 근거**다. 원본 로그의 개인 경로와 전체 내용은 공개 저장소에 넣지 않는다.

| 관측 | 의미 |
|---|---|
| SDPA 경고 뒤 recipe 28개 검사 완료 및 1~12/80 생성 완료 | 경고가 실행을 중단한 것이 아니다. 입력·recipe 사전 검사와 BF16 일부 생성까지 진행됐다 |
| D1-01~D1-04, D2-01~D2-02의 seed42/43, 모두 eos | 서로 다른 6문제의 12개 응답이 자연 종료했다고 보고됐다. 길이 307~2373토큰, 합계11198토큰 |
| 다음 model.generate 도중 ^C·KeyboardInterrupt | 사용자가 13번째 응답 중단. 고정 순서상 D2-03 seed42에 해당한다 |
| flash-attn 설치 metadata-generation-failed, psutil 없음 | 별도 설치 시도의 실패다. 현재 SDPA 실행에 flash-attn 설치는 필요하지 않다 |
| 같은 결과 폴더로 새 실행 시 FileExistsError | 기존 결과 보호 동작이다. 결과 폴더를 삭제해서 해결할 문제가 아니다 |
| 실행 파일 등을 rm한 뒤 파일 없음 | git pull만으로 최신 HEAD의 로컬 삭제가 자동 복구되지 않았다. 마지막 로그에서는 다시 FileExistsError가 있어 일부 복구 과정은 첨부에 보이지 않는다 |

**12회 EOS는 12회 정답 또는 난도 PASS를 뜻하지 않는다.** 최종 답을 보지 않았고 AWQ 생성은 아직 보고되지 않았다. 기존 R0의 128토큰 검열 결과와 이번 난도 시험의 자연 종료 보고는 별개의 실행 근거다.

## 재개 구현

기존 config·문제·정답·통과 기준·토큰 예산·sampling·모델 revision·SDPA·AWQ recipe·평가기 로직은 그대로 유지한다. 완료한 응답은 오답이나 length 종료여도 다시 생성하지 않는다. 결과를 바꾸기 위한 재시험은 하지 않는다.

- `--execute --resume`: 기존 폴더의 run.json을 검사하고, 정상 저장된 완료 응답을 건너뛴다. 이번 파일이 로그와 일치하면 **12개 보존 + 남은68개(BF16 28개, AWQ 40개)**다.
- 설정/문제/입력 ID·종료 필드·고정 순서·중복·패키지/Python·GPU·수치 실행 옵션·R0 및 원본 코드 참조를 대조한다. 공통 수치/평가 코드 해시도 바뀌면 중단한다. 실제 다시 로딩한 BF16 및 이미 생성 이력이 있는 AWQ의 전체 파라미터 해시가 이전 값과 같아야 한다.
- 재개 전 run.json의 원본 바이트를 `run.before_resume.<SHA256>.json`에 보존한다. 기존 응답 레코드와 최초 실행 provenance를 유지하고 재개 commit·코드 해시·보존 응답 해시·실행 구간을 별도로 기록한다.
- 같은 출력 폴더에서 두 프로세스가 동시에 쓰지 못하도록 파일 잠금을 사용한다. 기존 오류 응답이 있으면 자동 재시험하지 않고 검토를 요구한다.
- 새 코드는 Ctrl+C를 받으면 완료 응답을 보존하고 상태를 `DIFFICULTY_PILOT_INTERRUPTED_RESUMABLE`로 기록한다. 이전 코드는 KeyboardInterrupt를 잡지 못해 상태가 STARTED로 남을 수 있으므로 그 문자열만으로 실행 중이라고 판단하지 않는다.
- 가중치 해시 계산·GPU 이동·응답 시작을 표시한다. 생성 중에는 토큰이 진행되는 시점에 약15초 간격으로 진행 토큰 수를 출력한다. 이 표시용 추가 종료 조건은 항상 False이며 EOS나 토큰 상한을 바꾸지 않는다.

**중단된 응답의 저장되지 않은 토큰부터 이어가는 기능은 아니다.** 완료된12개는 그대로 쓰고 미완료13번째만 동일 문제·시드의 시작에서 다시 생성한다. 미완료 응답이 이미 소비한 토큰 수는 알 수 없고 실행 비용에서 사라진 것으로 취급하지 않는다. 남은68개 저장 대상 응답의 새 생성 상한은278528토큰이며, 실행 중 재개를 반복하면 버려진 부분 생성 비용은 별도로 늘어난다. 자동 재시도나 자동 예산 확대는 없다.

진행 표시가 추가된 재개 구간에는 그 비용이 포함된다. 과거12개와 새 응답을 섞은 시간 집계를 엄밀한 처리량 비교에 사용하지 않는다. 새 응답에는 execution_segment_index가 붙고 각 구간에 기록 방식 차이를 남긴다. 모델 답·종료의 검토와 성능 측정을 구분한다.

## 복구 후 재개 명령

사용자 GPU PC의 기존 저장소에서 실행한다. 먼저 **명시한 시험 파일 중 현재 삭제된 것만** 복구한다. 이미 존재하는 수정 파일·다른 파일·results 폴더는 건드리지 않는다. 별도 flash-attn 설치, 환경 재구축, R0/데이터 준비/합성 AWQ 재실행은 필요 없다.

```bash
conda activate quantthink
cd ~/quantthink
git diff --name-only --diff-filter=D -z -- \
  configs/difficulty_pilot_v01.json fixtures/difficulty_ladder_v01.json \
  'docs/DIFFICULTY_*_V01_KO.md' docs/REALISTIC_INPUT_PILOT_PLAN_20260929_KO.md \
  'scripts/*difficulty_pilot*.py' scripts/replay_r0_awq.py tests/test_difficulty_pilot.py \
  | xargs -0 -r git restore --source=HEAD --worktree --
git switch setup/session-01-research-gates &&
git pull --ff-only &&
python scripts/run_difficulty_pilot.py --execute --resume \
  --r0-dir results/local/r0_v03 \
  --upstream-dir results/local/awq_reference_v03 \
  --output-dir results/local/difficulty_v01 \
  --device cuda:0
```

원본 JSON이 첨부 로그와 같으면 “재개 확인: 완료12/80 보존, 남은68개” 이후 “시작13/80 bf16 D2-03 seed=42”가 표시된다. 파일·설정·가중치 해시 오류가 나면 결과를 지우거나 조건을 바꾸지 않고 그 오류를 검토한다. 파일이 실제로 몇 개의 완료 응답을 보존하고 있는지는 재개 명령에서 직접 확인한다.

완료 후 저장된 응답만 요약한다.

```bash
conda activate quantthink
cd ~/quantthink
python scripts/summarize_difficulty_pilot.py \
  --run results/local/difficulty_v01/run.json \
  --output results/local/difficulty_v01/review_after_resume.json
```

## 구현 검증과 연구 단계

새 `tests/test_difficulty_resume.py`의 **14개 CPU 검사 통과**. 12개→80개와 AWQ 부분 재개의 모의 실행에서 완료 응답이 보존되는지, 새 Ctrl+C 처리, 백업 바이트 보존, 동시 쓰기 차단, 조건/해시 불일치 거부, 읽기 전용 재개 조회, 진행 표시를 검증했다. 격리된 임시 Git 저장소에서 복구 명령이 삭제된 시험 파일만 복원하고 다른 수정·삭제·결과를 보존하는지도 확인했다.

모의 모델은 제어 흐름만 확인하며 실제 PyTorch/Transformers·CUDA·AWQ 수치 검사는 아니다. 새 실모델 실행은 이 환경에서 하지 않았다. 사용자 재개 성공·80회 결과·실제 최종 답 검토·좌표와 logit 검증은 아직 남아 있다. **model_ready=false와 세션02를 유지한다.**
