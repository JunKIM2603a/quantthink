#!/usr/bin/env python3
"""Resolve public HF metadata only; never download weights/data rows or run a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
MODEL_FILES = ("config.json", "generation_config.json", "tokenizer_config.json")


def metadata_url(asset: dict[str, Any]) -> str:
    repo_id = asset.get("repo_id", "")
    if not isinstance(repo_id, str) or not REPO.fullmatch(repo_id):
        raise ValueError("Invalid public repository id")
    kind = asset.get("repo_type")
    if kind not in ("model", "dataset"):
        raise ValueError("Only model/dataset metadata is supported")
    return f"https://huggingface.co/api/{'models' if kind == 'model' else 'datasets'}/{quote(repo_id, safe='/')}"


def read_json(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise ValueError("Only official HF HTTPS URLs are allowed")
    request = Request(url, headers={"User-Agent": "QuantThink-metadata-review/0.1", "Accept": "application/json"})
    # No token, credential, remote code, package installation or inference request.
    with urlopen(request, timeout=30) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.netloc != "huggingface.co":
            raise ValueError("Unexpected redirect host")
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Metadata exceeds the 2 MiB response limit")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def resolve_assets(manifest: dict[str, Any], fetch: Callable[[str], dict[str, Any]] = read_json) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for asset in manifest["assets"]:
        item = {"role": asset["role"], "repo_id": asset["repo_id"], "repo_type": asset["repo_type"]}
        try:
            info = fetch(metadata_url(asset))
            revision = info.get("sha")
            if not isinstance(revision, str) or not SHA.fullmatch(revision):
                raise ValueError("Missing or invalid immutable revision SHA")
            if info.get("id") != asset["repo_id"]:
                raise ValueError("Resolved id differs; review aliases explicitly")
            item.update({"revision": revision, "private": info.get("private"), "gated": info.get("gated"),
                         "metadata_canonical_sha256": canonical_hash(info), "status": "REVISION_RESOLVED"})
            if asset["repo_type"] == "model":
                files = {}
                for filename in MODEL_FILES:
                    # Read only three small public configuration files at the resolved commit.
                    url = f"https://huggingface.co/{asset['repo_id']}/resolve/{revision}/{filename}"
                    config = fetch(url)
                    files[filename] = {"source_url": url, "canonical_sha256": canonical_hash(config), "content": config}
                item["configuration_files"] = files
        except (HTTPError, URLError, OSError, ValueError, TypeError) as exc:
            item["status"] = "METADATA_FETCH_FAILED"
            error = {"role": asset["role"], "error_type": type(exc).__name__}
            if isinstance(exc, HTTPError):
                error["http_status"] = exc.code
            errors.append(error)
        assets.append(item)
    return {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_canonical_sha256": canonical_hash(manifest),
        "status": "METADATA_RESOLVED_NOT_RUN_READY" if not errors else "METADATA_INCOMPLETE",
        "assets": assets, "errors": errors,
        "scope": "Public metadata and three model configuration files only. No weights, dataset rows, model execution, approval or scientific gate pass.",
        "lock_status": "CANDIDATE_REVISIONS_REQUIRE_REVIEW_NOT_A_RUNTIME_LOCK",
    }


def write_new_report(path: Path, report: dict[str, Any]) -> None:
    # Serialize before opening so malformed payloads do not leave a partial file.
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/reproduction_candidate.json")
    parser.add_argument("--online", action="store_true", help="Explicitly allow bounded public metadata GET requests")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        assets = manifest["assets"]
        if not isinstance(assets, list) or not 1 <= len(assets) <= 8:
            raise ValueError("Expected 1..8 asset descriptors")
        roles = [a["role"] for a in assets]
        if len(set(roles)) != len(roles):
            raise ValueError("Duplicate asset roles")
        urls = [metadata_url(a) for a in assets]
        if not args.online:
            print(json.dumps({"status": "PLAN_ONLY_NO_NETWORK", "metadata_urls": urls,
                              "optional_model_files": list(MODEL_FILES)}, indent=2))
            return 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = args.output or ROOT / f"results/local/asset_revisions_{stamp}.json"
        if path.exists():
            raise FileExistsError("Choose a new output; no silent overwrite")
        report = resolve_assets(manifest)
        write_new_report(path, report)
        summary = {"status": report["status"], "errors": report["errors"],
                   "revisions": [{k: a.get(k) for k in ("role", "repo_id", "revision", "status")} for a in report["assets"]],
                   "scope": report["scope"]}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved: {path}")
        return 2 if report["errors"] else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot prepare metadata: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
