# QuantThink

양자화된 추론 모델의 추론 길이 증가와 정답 포기 현상을, 교란 크기를 맞춘 대조군으로 연구하는 프로젝트다.

**현재 상태: 합성 AWQ CUDA 점검 이후 데이터 준비에서 발생한 Pile 행 길이 오류를 재현·수정했다. 사용자 데이터 준비 재실행을 기다리며, 사전학습 BF16/AWQ 재현과 H1/H2 검정은 아직 실행하지 않았다.**

핵심 질문은 참조 모델의 공통 prefix에서 토큰 분포 KL을 맞춘 가우시안 가중치 교란이, 강한 양자화에서 관찰되는 길이 증가와 확인된 정답 포기를 재현하는지다. H2는 국소 엔트로피와 분기 행동에 관한 초안이다. 초기 모델은 DeepSeek-R1-Distill-Qwen-1.5B이며, 7B 재현과 MATH-500 확증 평가를 계획하고 있다.

## 먼저 읽을 문서

- [데이터 준비 중단 수정과 재실행 명령 — 한글](docs/DATA_PREPARATION_FIX_20260929_KO.md)
- [합성 AWQ 점검 검토와 다음 데이터 준비 — 한글](docs/SYNTHETIC_AWQ_REVIEW_20260929_KO.md)
- [재현 준비 v0.3 — 실제 AWQ 연결과 출처 검증](docs/REPRODUCTION_PREPARATION_V03_KO.md)
- [재현 준비 v0.2 — 좌표 검사·평가기·문제 분리 기록](docs/REPRODUCTION_PREPARATION_V02_KO.md)
- [토크나이저 계약 검토 — 한글](docs/TOKENIZER_CONTRACT_REVIEW_20260929.md)
- [현재 연구 상태](docs/research_status.json)
- [세션별 계획과 종료 조건](docs/SESSION_PLAN.md)
- [선행연구 본문 후속 검토](docs/NOVELTY_FOLLOWUP_20260929.md)
- [초기 근거·신규성 검토 기록](docs/EVIDENCE_AUDIT.md)
- [프로토콜 초안과 미결정 항목](docs/PROTOCOL_DRAFT.md)
- [세션 인계 양식](docs/HANDOFF_TEMPLATE.md)
- [연구 운영 규칙](AGENTS.md)

새로 전달하거나 갱신하는 안내 문서는 한국어로 작성한다. 과거 영문 문서는 검토 이력을 보존하기 위해 남겨 둔다.

가우시안 잡음 대조군이라는 발상 자체는 이미 활성 양자화 연구에 등장한다. 관련 논문 2609.23125v1과 2609.06473v1의 본문 후속 검토는 완료했으며, 최종 기여 범위 판정은 남아 있다. 평균 KL을 맞춘 대조군을 추가했다는 사실만으로 신규성을 주장하지 않는다.

## 현재 실행할 데이터 준비

기존 quantthink 환경에서 수정 사항과 새 회귀 검사를 확인한 뒤 데이터 준비를 다시 실행한다. pyarrow 20.0.0과 zstandard 0.23.0은 사용자 환경에 설치된 것으로 확인됐다.

~~~bash
conda activate quantthink
cd ~/quantthink
git switch setup/session-01-research-gates
git pull --ff-only

python -m unittest discover -s tests -p 'test_data_preparation_limits.py' -v

python scripts/prepare_runtime_assets.py \
  --mode all --online \
  --output-dir results/local/runtime_assets_v03
~~~

사용자 제출 단위 테스트 요약은 32개 중 31개 통과·1개 건너뜀이다. 합성 GPU 검사 상태는 SYNTHETIC_AWQ_ADAPTER_PASS_NOT_MODEL_READY이며, 제출된 구현 해시 4개와 정책 해시가 당시 검토 commit 4076739와 일치했다. 원본 로컬 보고서 바이트나 생성 commit까지 독립 검증한 것은 아니다.

앞선 작성 환경의 v0.3 검사 29개 통과·3개 건너뜀과 v0.2 연계 검사 32개 통과는 별도의 과거 이력이다. SDPA 경고의 조건과 512토큰 범위 해석은 최신 한글 검토 문서에 기록했다. 긴 문맥이나 실모델 동등성의 성공으로 확대 해석하지 않는다.

위 명령은 고정 데이터 파일의 해시와 개발 100문제·calibration 65,536토큰 후보를 준비한다. 모델 가중치나 생성은 사용하지 않는다. 성공 상태는 PREPARED_CANDIDATE_NOT_RUN_APPROVAL이며, 결과를 공유할 요약 명령은 최신 한글 검토 문서에 있다. 사용자 데이터 준비는 행 길이 한도로 중단됐다. 같은 고정 Pile 파일의 최대 행 4,744,757자가 기존 한도를 초과함을 확인해 calibration 읽기 한도를 8,388,608자로 수정했다. zstandard 0.23.0에서 실제 179,996행 전체 JSON 해석을 확인했고, 전체 데이터 후보 생성은 재실행 대기다.

사용자가 제출한 토크나이저·합성 AWQ 검사는 반복할 필요가 없다. 설치된 데이터 준비 패키지와 다운로드 캐시를 사용한다. 수정에 따른 runtime_assets.py의 새 코드 해시는 다음 준비 보고서에 기록된다.

기존 환경 진단 도구 scripts/preflight.py는 별도 환경 점검용으로 유지한다. 보고서는 results/local에 기록되며 Git에서 제외된다. CUDA_VISIBLE이나 작은 BF16 행렬 계산 성공은 모델·AWQ·장문맥 실행 성공을 뜻하지 않는다.

## 결과 해석 원칙

긴 응답, 반복, 상한 도달, 올바른 중간 풀이를 버린 현상은 서로 구분한다. 평균 KL이 같아도 교란의 구조나 EOS 확률이 같다는 보장은 없다. 동등성을 입증하지 못한 결과를 곧바로 양자화 고유 메커니즘의 증거로 해석하지 않는다.

가짜 양자화 후 BF16 실행은 native 저비트 처리량 측정이 아니다. 가설·동등성 경계·실험 설정은 아직 동결하지 않았으며, 모델 실험 전에 프로토콜과 실행 조건을 기록해야 한다.
