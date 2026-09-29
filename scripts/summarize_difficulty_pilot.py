"""난도 시험 run.json을 읽어 기존 생성만 평가합니다. 모델·토크나이저·네트워크 사용 없음."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from difficulty_pilot_contracts import file_hash, load_suite, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        cfg, suite = load_suite()
        report = json.loads(args.run.read_text(encoding="utf-8"))
        summary = summarize(report, cfg, suite)
        summary["run_file_sha256"] = file_hash(args.run)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        for row in summary["levels"]:
            print(f"D{row['level']} {row['arm']:14s} {row['verdict']:12s} "
                  f"안정통과={row['stable_problems']}/4 EOS={row['eos_rate_over_planned']:.0%} "
                  f"검열={row['censored_rate_over_planned']:.0%}")
        print(json.dumps(summary["boundaries"], ensure_ascii=False, indent=2))
        print(f"생성 기록 {summary['observed_attempts']}/{summary['expected_attempts']}; 모델 준비/동등성 인증 아님.")
        return 0
    except Exception as exc:
        parser.exit(2, f"요약 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
