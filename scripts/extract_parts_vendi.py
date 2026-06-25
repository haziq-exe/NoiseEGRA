#!/usr/bin/env python3
"""Extract Beginning/Middle/End story parts for Vendi analysis.

Workflow:
1) For each score file in experiment_results/SCORES/*_SCORE.csv,
2) Keep stories where Structure == 1,
3) Ask Azure OpenAI to choose strict split boundaries,
4) Slice exact text from original stories into Beginning/Middle/End,
5) Save per-run CSV outputs under experiment_results/PARTS_VENDI.

No embedding or vendi computation is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_common import chat_with_retry, create_azure_client, repo_root
from scoring_common import resolve_results_dir


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _as_int(v: object) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return 0


def read_stories(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    out: List[str] = []
    for r in rows:
        if not r:
            continue
        s = " ".join(part.strip() for part in r if str(part).strip())
        if s:
            out.append(s)
    if not out:
        raise ValueError(f"No stories found in {path}")
    return out


def iter_source_csvs(egra_results: Path) -> Iterable[Path]:
    for p in sorted(egra_results.rglob("*.csv")):
        if "SCORES" in p.parts:
            continue
        if "PARTS_VENDI" in p.parts:
            continue
        yield p


def build_source_maps(egra_results: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    by_stem: Dict[str, Path] = {}
    by_norm: Dict[str, List[Path]] = {}
    for p in iter_source_csvs(egra_results):
        by_stem[p.stem] = p
        by_norm.setdefault(_norm(p.stem), []).append(p)
    return by_stem, by_norm


def find_source_csv(score_stem: str, by_stem: Dict[str, Path], by_norm: Dict[str, List[Path]]) -> Optional[Path]:
    if score_stem in by_stem:
        return by_stem[score_stem]
    cands = by_norm.get(_norm(score_stem), [])
    if len(cands) == 1:
        return cands[0]
    return None


@dataclass(frozen=True)
class StoryTarget:
    story_number: int  # 1-based
    story_text: str


@dataclass(frozen=True)
class StoryParts:
    story_number: int
    beginning: str
    middle: str
    end: str


def _thirds_fallback(story_number: int, story: str) -> StoryParts:
    n = len(story)
    if n < 3:
        # Degenerate case: force minimal non-empty partition with duplication-avoidant slices.
        b = story[:1]
        m = story[1:2] if n > 1 else " "
        e = story[2:] if n > 2 else " "
        return StoryParts(story_number=story_number, beginning=b, middle=m, end=e)

    begin_end = max(1, n // 3)
    middle_end = max(begin_end + 1, (2 * n) // 3)
    if middle_end >= n:
        middle_end = n - 1

    return StoryParts(
        story_number=story_number,
        beginning=story[:begin_end],
        middle=story[begin_end:middle_end],
        end=story[middle_end:],
    )


def _is_content_filter_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "content_filter" in s or "responsibleaipolicyviolation" in s


def collect_structure_one_targets(score_csv: Path, stories: Sequence[str]) -> List[StoryTarget]:
    targets: List[StoryTarget] = []
    with score_csv.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        row_idx = 0
        for row in reader:
            row_idx += 1
            story_number = _as_int(row.get("Story number")) or row_idx
            structure = _as_int(row.get("Structure"))
            if structure != 1:
                continue
            if 1 <= story_number <= len(stories):
                targets.append(StoryTarget(story_number=story_number, story_text=stories[story_number - 1]))
    return targets


def _extract_json(text: str) -> dict:
    t = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        t = fenced.group(1).strip()

    try:
        data = json.loads(t)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        data = json.loads(t[l : r + 1])
        if isinstance(data, dict):
            return data

    raise ValueError("Model did not return parseable JSON object")


def _segment_batch(
    client,
    model: str,
    batch: Sequence[StoryTarget],
    max_retries: int = 3,
) -> List[StoryParts]:
    retry_note = ""

    for _attempt in range(max_retries):
        payload = [
            {
                "story_number": t.story_number,
                "story": t.story_text,
            }
            for t in batch
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict text segmentation tool. "
                    "You MUST NOT rewrite, summarize, or invent any text. "
                    "Return ONLY JSON. "
                    "For each input story, return exactly two integer split indices using Python 0-based slicing. "
                    "Definitions: beginning=story[0:begin_end], middle=story[begin_end:middle_end], end=story[middle_end:]. "
                    "Rules: 0 < begin_end < middle_end < len(story), and all three parts must be non-empty. "
                    "Each part must be substantial (not tiny 1-2 character fragments). "
                    "Choose boundaries near natural narrative transitions when possible."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Input JSON array (story_number + story text):\n"
                    f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                    "Output format exactly:\n"
                    "{\"items\":[{\"story_number\":<int>,\"begin_end\":<int>,\"middle_end\":<int>}]}\n"
                    "Include one item per input story_number, same order."
                    + ("\n\nPrevious attempt errors to fix: " + retry_note if retry_note else "")
                ),
            },
        ]

        content = chat_with_retry(client, model=model, messages=messages)

        try:
            parsed = _extract_json(content)
            items = parsed.get("items")
            if not isinstance(items, list):
                raise ValueError("JSON missing list field 'items'")
            if len(items) != len(batch):
                raise ValueError(f"Expected {len(batch)} items, got {len(items)}")

            out: List[StoryParts] = []
            for target, item in zip(batch, items):
                if not isinstance(item, dict):
                    raise ValueError("Each item must be an object")

                story_number = _as_int(item.get("story_number"))
                begin_end = _as_int(item.get("begin_end"))
                middle_end = _as_int(item.get("middle_end"))

                story = target.story_text
                n = len(story)
                if story_number != target.story_number:
                    raise ValueError(
                        f"story_number mismatch: expected {target.story_number}, got {story_number}"
                    )
                if not (0 < begin_end < middle_end < n):
                    raise ValueError(
                        f"Invalid boundaries for story {story_number}: begin_end={begin_end}, middle_end={middle_end}, len={n}"
                    )

                beginning = story[:begin_end]
                middle = story[begin_end:middle_end]
                end = story[middle_end:]
                if not beginning or not middle or not end:
                    raise ValueError(f"Empty segment for story {story_number}")

                # Prevent degenerate boundaries; keep each section substantial.
                min_seg = max(20, min(120, n // 10))
                if len(beginning) < min_seg or len(middle) < min_seg or len(end) < min_seg:
                    raise ValueError(
                        f"Segments too short for story {story_number}: "
                        f"len(b,m,e)=({len(beginning)},{len(middle)},{len(end)}), min={min_seg}"
                    )

                out.append(
                    StoryParts(
                        story_number=story_number,
                        beginning=beginning,
                        middle=middle,
                        end=end,
                    )
                )

            return out
        except Exception as exc:
            retry_note = str(exc)
            continue

    raise RuntimeError(f"Failed to segment batch after {max_retries} attempts: {retry_note}")


def extract_parts_for_run(
    client,
    model: str,
    source_csv: Path,
    score_csv: Path,
    targets: Sequence[StoryTarget],
    batch_size: int,
) -> List[StoryParts]:
    parts: List[StoryParts] = []
    queue: List[List[StoryTarget]] = [list(targets[i : i + batch_size]) for i in range(0, len(targets), batch_size)]

    while queue:
        batch = queue.pop(0)
        try:
            parts.extend(_segment_batch(client=client, model=model, batch=batch))
        except Exception as exc:
            if len(batch) > 1:
                mid = len(batch) // 2
                # Split and retry to isolate problematic stories.
                queue.insert(0, batch[mid:])
                queue.insert(0, batch[:mid])
                continue

            # Single-story batch: use deterministic exact-slice fallback.
            t = batch[0]
            parts.append(_thirds_fallback(story_number=t.story_number, story=t.story_text))
            if _is_content_filter_error(exc):
                continue
            continue

    parts.sort(key=lambda x: x.story_number)
    return parts


def write_parts_csv(out_path: Path, source_csv: Path, score_csv: Path, parts: Sequence[StoryParts]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Story number", "Beginning", "Middle", "End", "SourceCSV", "ScoreCSV"])
        for p in parts:
            w.writerow([
                p.story_number,
                p.beginning,
                p.middle,
                p.end,
                str(source_csv.relative_to(repo_root())),
                str(score_csv.relative_to(repo_root())),
            ])


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract Beginning/Middle/End parts for Structure==1 stories")
    ap.add_argument("--results-dir", type=Path, default=None, help="Experiment results root")
    ap.add_argument("--batch-size", type=int, default=8, help="Stories per model call")
    ap.add_argument(
        "--only-stem",
        action="append",
        default=[],
        help="Process only this run stem (without _SCORE). Can be repeated.",
    )
    args = ap.parse_args()

    root = repo_root()
    egra_results = resolve_results_dir(args.results_dir)
    scores_dir = egra_results / "SCORES"
    parts_dir = egra_results / "PARTS_VENDI"

    client, model = create_azure_client()

    by_stem, by_norm = build_source_maps(egra_results)
    score_files = sorted(scores_dir.glob("*_SCORE.csv"))

    if args.only_stem:
        allowed = {_norm(s) for s in args.only_stem}
        filtered: List[Path] = []
        for p in score_files:
            stem = p.stem[:-len("_SCORE")] if p.stem.endswith("_SCORE") else p.stem
            if _norm(stem) in allowed:
                filtered.append(p)
        score_files = filtered

    parts_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: List[str] = []
    summary_lines.append(f"Model: {model}")
    summary_lines.append(f"Score files scanned: {len(score_files)}")

    total_targets = 0
    total_extracted = 0

    for score_csv in score_files:
        score_stem = score_csv.stem
        if score_stem.endswith("_SCORE"):
            score_stem = score_stem[: -len("_SCORE")]

        source_csv = find_source_csv(score_stem, by_stem, by_norm)
        if source_csv is None:
            summary_lines.append(f"SKIP {score_csv.name}: source CSV not found")
            continue

        stories = read_stories(source_csv)
        targets = collect_structure_one_targets(score_csv=score_csv, stories=stories)
        total_targets += len(targets)

        rel_parent = source_csv.parent.relative_to(egra_results)
        out_path = parts_dir / rel_parent / f"{source_csv.stem}__PARTS.csv"

        if not targets:
            write_parts_csv(out_path=out_path, source_csv=source_csv, score_csv=score_csv, parts=[])
            summary_lines.append(f"{source_csv.stem}: Structure=1 stories 0 -> wrote empty {out_path.relative_to(root)}")
            print(f"[DONE] {source_csv.stem}: no Structure=1 stories")
            continue

        print(f"[RUN] {source_csv.stem}: extracting parts for {len(targets)} stories")
        try:
            parts = extract_parts_for_run(
                client=client,
                model=model,
                source_csv=source_csv,
                score_csv=score_csv,
                targets=targets,
                batch_size=args.batch_size,
            )
            write_parts_csv(out_path=out_path, source_csv=source_csv, score_csv=score_csv, parts=parts)

            total_extracted += len(parts)
            summary_lines.append(
                f"{source_csv.stem}: Structure=1 stories {len(targets)} -> {out_path.relative_to(root)}"
            )
            print(f"[DONE] {source_csv.stem}: wrote {out_path.relative_to(root)}")
        except Exception as exc:
            summary_lines.append(f"ERROR {source_csv.stem}: {exc}")
            print(f"[ERROR] {source_csv.stem}: {exc}")

    summary_lines.append(f"Total Structure=1 targets: {total_targets}")
    summary_lines.append(f"Total extracted parts: {total_extracted}")

    manifest = parts_dir / "MANIFEST.txt"
    manifest.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
