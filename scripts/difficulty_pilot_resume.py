"""완료 응답 보존·재개 계약과 진행 표시. 모델을 로드하지 않는 표준 라이브러리 코드."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time

from difficulty_pilot_contracts import canonical_hash, summarize


def attempt_key(row):
    return row["problem_id"], row["arm"], row["seed"]


def planned_order(cfg, suite):
    return [(p["id"], arm, seed) for arm in cfg["arms"]
            for p in suite["problems"] for seed in cfg["seeds"]]


def resume_plan(report, cfg, suite):
    """이미 저장된 eos/length는 정오와 무관하게 보존. 오류 응답 자동 재시험 없음."""
    summarize(report, cfg, suite)  # 종료 ID, 입력/설정/문제 해시, 중복도 먼저 검사합니다.
    rows = report["attempts"]
    if any(row.get("finish_reason") not in {"eos", "length"} for row in rows):
        raise ValueError("저장된 실행 오류가 있습니다. 자동 재시험하지 말고 오류를 먼저 검토하세요.")
    order = planned_order(cfg, suite)
    keys = [attempt_key(row) for row in rows]
    if keys != order[:len(keys)]:
        raise ValueError("저장된 완료 응답이 고정 실행 순서의 연속 구간과 다릅니다.")
    for arm in {r["arm"] for r in rows}:
        digest = report.get("parameter_sha256", {}).get(arm)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("완료 응답에 대응하는 모델 파라미터 해시가 없습니다.")
    return {"completed": len(keys), "remaining": len(order) - len(keys),
            "completed_keys": keys, "pending_keys": order[len(keys):],
            "completed_attempts_sha256": canonical_hash(rows)}


def load_resume(path, cfg, suite):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("재개 원본 run.json은 심볼릭 링크일 수 없습니다.")
    report = json.loads(path.read_text(encoding="utf-8"))
    return report, resume_plan(report, cfg, suite)


def verify_compatibility(previous, current, inputs):
    """진행/재개 코드 변경만 허용하고 모델·디코딩·공통 수치 경로는 비교합니다."""
    fields = ("config_sha256", "suite_sha256", "packages", "python", "model_reference",
              "r0_input_files", "upstream_source_files", "generation_config", "runtime_flags", "gpu")
    for key in fields:
        if previous.get(key) != current.get(key):
            raise ValueError(f"재개 전후 실행 조건 불일치: {key}")
    critical = ("difficulty_pilot_contracts.py", "replay_r0_awq.py", "qwen2_awq_adapter.py",
                "runtime_assets.py", "runtime_contracts.py", "reproduction_contracts.py")
    for key in critical:
        if previous["implementation_sha256"].get(key) != current["implementation_sha256"].get(key):
            raise ValueError(f"재개 전후 공통 구현 해시 불일치: {key}")
    for row in previous["attempts"]:
        if row["input_ids"] != inputs[row["problem_id"]]:
            raise ValueError("재개 시 토크나이저가 만든 입력 ID가 기존 응답과 다릅니다.")


def check_parameter_hash(report, arm, observed):
    expected = report["parameter_sha256"].get(arm)
    if expected is not None and observed != expected:
        raise ValueError(f"재구성/재로딩한 {arm} 파라미터 해시가 이전 실행과 다릅니다.")
    report["parameter_sha256"][arm] = observed


def backup_before_resume(path):
    """원본 바이트는 내용 해시 이름으로 보존하며 기존 백업을 덮어쓰지 않습니다."""
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    backup = path.with_name(f"run.before_resume.{digest}.json")
    try:
        with backup.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if backup.read_bytes() != data:
            raise ValueError("기존 백업 내용이 예상과 다릅니다.")
    return {"file": backup.name, "sha256": digest}


@contextmanager
def output_lock(folder, *, resume):
    folder = Path(folder)
    if resume:
        if not (folder / "run.json").is_file():
            raise ValueError("재개할 run.json이 없습니다. 기존 결과 경로를 확인하세요.")
    else:
        try:
            folder.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError("출력 폴더가 이미 있습니다. 결과를 삭제하지 말고 --execute --resume으로 이어가세요.") from exc
    with (folder / ".difficulty_pilot.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("같은 결과 폴더에서 다른 난도 시험이 실행 중입니다.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def mark_interrupted(report):
    active = report.pop("active_attempt", None)
    report.setdefault("interruptions", []).append({
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_attempts": len(report["attempts"]), "active_attempt": active,
        "partial_generated_ids_saved": False,
        "note": "미완료 응답의 토큰/RNG 상태는 보존하지 않음. 명시적 재개 시 이 응답만 시작부터 생성."})
    report["status"] = "DIFFICULTY_PILOT_INTERRUPTED_RESUMABLE"


class ProgressReporter:
    """토큰을 읽거나 RNG/종료 판단을 바꾸지 않는 15초 간격 표시."""
    def __init__(self, label, cap, *, clock=time.monotonic, emit=print, interval=15):
        self.label, self.cap = label, cap
        self.clock, self.emit, self.interval = clock, emit, interval
        self.started = self.last = clock()

    def tick(self, generated_tokens):
        now = self.clock()
        if now - self.last >= self.interval:
            self.emit(f"진행 {self.label}: {generated_tokens}/{self.cap}토큰, {now - self.started:.0f}초", flush=True)
            self.last = now
