"""고정 revision의 파일 검증과 데이터 준비. 모델 가중치를 읽거나 추론하지 않습니다."""
from __future__ import annotations

import hashlib
import heapq
import importlib.metadata
import io
import json
from pathlib import Path
import re
from urllib.request import urlopen

from build_development_manifest import (
    build_manifest, canonical_hash, normalized_question, problem_id, question_hash, write_new,
)
from evaluate_gsm8k import gold_value

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/runtime_preparation_v03.json"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_JSONL_MAX_LINE_CHARACTERS = 4 * 1024 * 1024
# 고정 Pile validation 전체 검사: 최대 4,744,757자. 표본 선택 한도가 아닌 파서 자원 한도입니다.
CALIBRATION_JSONL_MAX_LINE_CHARACTERS = 8 * 1024 * 1024


def load_contracts():
    candidate = json.loads((ROOT / "configs/reproduction_candidate.json").read_text())
    refs = json.loads((ROOT / "configs/asset_inspection_refs.json").read_text())
    policy = json.loads(POLICY_PATH.read_text())
    if canonical_hash(candidate) != policy["base_manifest_sha256"]:
        raise ValueError("후보 설정의 정규화 해시가 변경됐습니다.")
    if canonical_hash(refs) != policy["inspection_refs_sha256"]:
        raise ValueError("점검 참조의 정규화 해시가 변경됐습니다.")
    for asset in refs["assets"]:
        if not HEX40.fullmatch(asset["revision"]):
            raise ValueError("전체 40자리 고정 revision이 필요합니다.")
    return candidate, refs, policy


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def implementation_fingerprint(names):
    return {name: file_digests(ROOT / "scripts" / name)["sha256"] for name in names}


def file_digests(path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(chunk)
            sha256.update(chunk)
            blob.update(chunk)
    if count != size:
        raise ValueError("파일 크기가 검사 중 달라졌습니다.")
    return {"size": size, "sha256": sha256.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def source_plan(info, asset, filename):
    """Hub repo_info(files_metadata=True)의 한 파일을 검증 가능한 계획으로 변환합니다."""
    if info.sha != asset["revision"]:
        raise ValueError("요청한 revision과 Hub 응답이 다릅니다.")
    siblings = [x for x in info.siblings if x.rfilename == filename]
    if len(siblings) != 1:
        raise ValueError("고정 경로의 원본 파일이 없거나 중복됐습니다.")
    item = siblings[0]
    if type(item.size) is not int or item.size <= 0:
        raise ValueError("원본 파일 크기 메타데이터가 없습니다.")
    if item.lfs is not None:
        digest = item.lfs.sha256
        if not HEX64.fullmatch(digest) or item.lfs.size != item.size:
            raise ValueError("LFS 해시·크기 메타데이터가 올바르지 않습니다.")
        algorithm = "sha256"
    else:
        digest = item.blob_id
        if not isinstance(digest, str) or not HEX40.fullmatch(digest):
            raise ValueError("Git blob 해시 메타데이터가 없습니다.")
        algorithm = "git_blob_sha1"
    return {**asset, "path": filename, "size": item.size,
            "digest_algorithm": algorithm, "expected_digest": digest}


def plan_sources(api, refs, policy, roles):
    plans = []
    for role in roles:
        matches = [x for x in refs["assets"] if x["role"] == role]
        if len(matches) != 1:
            raise ValueError("출처 역할을 하나로 지정해야 합니다.")
        asset = matches[0]
        info = api.repo_info(repo_id=asset["repo_id"], repo_type=asset["repo_type"],
                             revision=asset["revision"], files_metadata=True,
                             timeout=30, token=False)
        plans.append(source_plan(info, asset, policy["source_paths"][role]))
    if sum(p["size"] for p in plans) > policy["max_dataset_download_bytes"]:
        raise ValueError("전체 다운로드 계획이 크기 상한을 넘었습니다. 자동 다운로드하지 않습니다.")
    return plans


def verify_source(path, plan):
    observed = file_digests(path)
    if (observed["size"] != plan["size"]
            or observed[plan["digest_algorithm"]] != plan["expected_digest"]):
        raise ValueError("원본 파일 바이트와 고정 revision의 원격 해시가 다릅니다.")
    return {**plan, "observed_sha256": observed["sha256"],
            "verification": "BYTES_MATCH_PINNED_HUB_METADATA"}


def download_sources(plans, downloader):
    paths, evidence = {}, []
    for plan in plans:
        path = Path(downloader(repo_id=plan["repo_id"], repo_type=plan["repo_type"],
                               filename=plan["path"], revision=plan["revision"],
                               endpoint="https://huggingface.co", token=False))
        proof = verify_source(path, plan)
        paths[plan["role"]] = path
        evidence.append(proof)
    return paths, evidence


def iter_jsonl(path, *, compressed=False, max_rows=1_000_000,
               max_line_characters=DEFAULT_JSONL_MAX_LINE_CHARACTERS, statistics=None):
    """실제 행 번호와 전체 JSON을 보존합니다. 자원 한도를 넘으면 원인을 구분해 중단합니다."""
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError("JSONL max_rows는 양의 정수여야 합니다.")
    if type(max_line_characters) is not int or max_line_characters < 1:
        raise ValueError("JSONL max_line_characters는 양의 정수여야 합니다.")
    path = Path(path)
    stats = statistics if statistics is not None else {}
    stats.update(max_rows=max_rows, max_line_characters=max_line_characters,
                 source_rows=0, largest_line_characters=0,
                 rows_above_default_line_limit=0,
                 length_unit="Unicode characters including line terminator")

    def rows(stream):
        for index in range(max_rows + 1):
            raw = stream.readline(max_line_characters + 1)
            if not raw:
                return
            location = f"{path.name}, source_row={index} (행 {index + 1})"
            if index == max_rows:
                raise ValueError(f"JSONL 행 수 상한 초과: {location}, max_rows={max_rows}")
            if len(raw) > max_line_characters:
                raise ValueError(f"JSONL 행 길이 상한 초과: {location}, "
                                 f"최소 {len(raw)}자 > {max_line_characters}자 (줄바꿈 포함)")
            if not raw.strip():
                raise ValueError(f"원본 JSONL에 빈 행이 있습니다: {location}")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 구문 오류: {location}, {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL 행은 객체여야 합니다: {location}")
            stats["source_rows"] += 1
            stats["largest_line_characters"] = max(stats["largest_line_characters"], len(raw))
            stats["rows_above_default_line_limit"] += int(len(raw) > DEFAULT_JSONL_MAX_LINE_CHARACTERS)
            yield index, obj
    if compressed:
        import zstandard
        with path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    yield from rows(text)
    else:
        with path.open(encoding="utf-8") as text:
            yield from rows(text)


def read_development(path):
    import pyarrow.parquet as parquet
    return parquet.read_table(path, columns=["question", "answer"]).to_pylist()


def development_bundle(development, confirmation, candidate, policy):
    if len(development) != policy["expected_development_rows"]:
        raise ValueError("GSM8K main/train 전체 행 수가 예상과 다릅니다.")
    manifest = build_manifest(development, confirmation, n=candidate["r1"]["n_problems"],
                              seed=candidate["r1"]["selection_seed"],
                              expected_confirmation_rows=policy["expected_confirmation_rows"])
    by_id = {}
    for index, row in enumerate(development):
        key = problem_id(row["question"])
        numeric = gold_value(row["answer"])
        if key in by_id and gold_value(by_id[key]["answer"]) != numeric:
            raise ValueError("정규화된 동일 문제에 상충하는 원본 정답이 있습니다.")
        if key not in by_id:
            by_id[key] = {"id": key, "question": normalized_question(row["question"]),
                          "answer": row["answer"], "source_row": index}
    chosen = [by_id[x["id"]] for x in manifest["selected"]]
    manifest["manifest_canonical_sha256"] = canonical_hash(candidate)
    manifest["preparation_policy_sha256"] = canonical_hash(policy)
    inputs = [{k: row[k] for k in ("id", "question")} for row in chosen]
    gold = [{k: row[k] for k in ("id", "answer")} for row in chosen]
    manifest["source_row_indices"] = [row["source_row"] for row in chosen]
    return manifest, inputs, gold


def select_calibration(rows, encode, excluded_questions, policy):
    """짧은 적격 문서의 해시 순서를 고정하고 정확한 토큰 수로 자릅니다."""
    cfg = policy["calibration"]
    pool, seen = [], set()
    counts = {"source_rows": 0, "eligible_unique_documents": 0,
              "excluded_exact_question_overlap": 0, "duplicate_documents": 0,
              "excluded_empty_text": 0, "excluded_text_length": 0,
              "excluded_token_length": 0}
    for row_index, row in rows:
        counts["source_rows"] += 1
        if type(row_index) is not int or row_index < 0:
            raise ValueError("calibration 원본 행 번호가 유효하지 않습니다.")
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("calibration text 필드가 없습니다.")
        text = text.strip()
        if not text:
            counts["excluded_empty_text"] += 1
            continue
        if len(text) > cfg["max_text_characters"]:
            counts["excluded_text_length"] += 1
            continue
        if question_hash(text) in excluded_questions:
            counts["excluded_exact_question_overlap"] += 1
            continue
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in seen:
            counts["duplicate_documents"] += 1
            continue
        seen.add(text_hash)
        ids = list(encode(text))
        if any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("토크나이저 출력 ID가 유효하지 않습니다.")
        if not 0 < len(ids) <= cfg["max_document_tokens"]:
            counts["excluded_token_length"] += 1
            continue
        counts["eligible_unique_documents"] += 1
        rank = int(hashlib.sha256(f"{cfg['seed']}\n{row_index}".encode()).hexdigest(), 16)
        record = {"source_row": row_index, "text_sha256": text_hash, "ids": ids}
        item = (-rank, -row_index, record)
        if len(pool) < cfg["pool_documents"]:
            heapq.heappush(pool, item)
        elif item[:2] > pool[0][:2]:
            heapq.heapreplace(pool, item)
    needed = cfg["blocks"] * cfg["block_size"]
    flat, documents = [], []
    for _, _, row in sorted(pool, key=lambda x: (-x[0], -x[1])):
        used = row["ids"][:needed - len(flat)]
        if not used:
            break
        flat.extend(used)
        documents.append({"source_row": row["source_row"], "text_sha256": row["text_sha256"],
                          "original_tokens": len(row["ids"]), "used_tokens": len(used),
                          "used_token_ids_sha256": canonical_hash(used)})
    if len(flat) != needed:
        raise ValueError("고정 calibration 후보 풀에서 필요한 토큰 수를 채우지 못했습니다.")
    blocks = [flat[i:i + cfg["block_size"]] for i in range(0, needed, cfg["block_size"])]
    return blocks, {"policy": cfg, "counts": counts, "documents": documents,
                    "token_blocks_sha256": canonical_hash(blocks),
                    "total_tokens": needed, "used_documents": len(documents)}


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def verify_upstream(folder, policy):
    records = []
    for entry in policy["upstream"]["files"]:
        path = folder / entry["path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError("고정 원본 파일이 없거나 심볼릭 링크입니다.")
        actual = file_digests(path)
        if actual["git_blob_sha1"] != entry["git_blob_sha1"]:
            raise ValueError(f"AWQ 원본 해시 불일치: {entry['path']}")
        records.append({**entry, "sha256": actual["sha256"]})
    return records


def fetch_upstream(folder, policy, *, opener=urlopen):
    """필요한 원본 모듈 5개와 LICENSE만 받으며 아직 실행하지 않습니다."""
    folder.mkdir(parents=True, exist_ok=True)
    source = policy["upstream"]
    for entry in source["files"]:
        target = folder / entry["path"]
        if target.exists():
            if target.is_symlink() or file_digests(target)["git_blob_sha1"] != entry["git_blob_sha1"]:
                raise ValueError("기존 AWQ 원본 파일의 해시가 다릅니다. 덮어쓰지 않습니다.")
            continue
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{entry['path']}"
        with opener(url, timeout=30) as response:
            data = response.read(512 * 1024 + 1)
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) > 512 * 1024 or digest != entry["git_blob_sha1"]:
            raise ValueError("다운로드한 AWQ 원본이 고정 blob과 일치하지 않습니다.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as out:
            out.write(data)
    return verify_upstream(folder, policy)
