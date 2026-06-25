"""Batch-score Arabic stories in experiment_results CSV files.

Input files are treated as one story per row with no header.
Outputs are written to experiment_results/SCORES with the same base name + _SCORE.csv.

Uses Azure OpenAI via the `openai` Python package.
Credentials are loaded from environment variables and `.env` (see .env.example).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_common import chat_with_retry, create_azure_client, repo_root
from scoring_common import resolve_results_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noiseegra.defaults import DEFAULT_EXPERIMENT_INPUT_FOLDERS

DEFAULT_INPUT_FOLDERS = DEFAULT_EXPERIMENT_INPUT_FOLDERS

OUTPUT_COLUMNS = [
    "Story number",
    "Readability",
    "Logic",
    "GrammarandLinguistics",
    "ReadingLevel",
    "TotalModalCollapse",
    "Structure",
    "VocabularyLevel",
    "Stereotypes",
    "Gender-balanced",
]

PROMPT = (
    "Attached is a csv file of Arabic stories. I want you to rate each story out of 10 on the following metrics: \n"
    "- Readability: How well does the story read, is it just a set of weakly connected sentences or does it flow well etc. \n"
    "- Logic: How much does the story make sense, does it have logical fallacies etc. - Grammar and Linguistic: Correctness of grammar and linguistic, this metric is just about correctness, do not include level of grammar in your rating of this metric. Here are other metrics to include also \n"
    "- Reading Level: What grade level is this story appropriate for? \n"
    "- Total Modal Collapse: If this story indicates total modal collapse, give zero on every other metrics and output a 1 here, if not then leave as zero \n"
    "- Structure: Does the narrative structure includes intro, middle dilemma, and ending with resolution. 1 if yes and 0 if no \n"
    "- Vocabulary level: Vocabulary suitable for children and local context or not, 1 if yes and 0 if not \n"
    "- Stereotypes: Avoids gender/religion/other stereotypes. 1 if yes and 0 if not \n"
    "- Gender-balanced: includes both a boy and a girl. 1 if yes and 0 if not\n\n"
    "Your response should be as a table format csv with the following column names:\n\n"
    "Story number, Readability, Logic, GrammarandLinguistics, ReadingLevel, TotalModalCollapse, Structure, VocabularyLevel, Stereotypes, Gender-balanced"
)


@dataclass(frozen=True)
class ScoreRow:
    story_number: int
    readability: float
    logic: float
    grammar_and_linguistics: float
    reading_level: str
    total_modal_collapse: int
    structure: int
    vocabulary_level: int
    stereotypes: int
    gender_balanced: int

    def to_csv_row(self) -> List[str]:
        return [
            str(self.story_number),
            str(self.readability),
            str(self.logic),
            str(self.grammar_and_linguistics),
            self.reading_level,
            str(self.total_modal_collapse),
            str(self.structure),
            str(self.vocabulary_level),
            str(self.stereotypes),
            str(self.gender_balanced),
        ]


def iter_input_csv_files(egra_results: Path, input_folders: Sequence[str]) -> Iterable[Path]:
    for folder in input_folders:
        folder_path = egra_results / folder
        if not folder_path.is_dir():
            continue
        for path in sorted(folder_path.glob("*.csv")):
            if path.name.endswith("_SCORE.csv"):
                continue
            yield path


def read_stories(path: Path) -> List[str]:
    """Read one story per CSV row (stories may contain embedded newlines)."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))

    stories: List[str] = []
    for r in rows:
        if not r:
            continue
        s = " ".join(part.strip() for part in r if str(part).strip())
        if s:
            stories.append(s)

    if not stories:
        raise ValueError(
            f"No stories found in {path}. Expected a single-column CSV with one story per row."
        )
    return stories


def _extract_csv_block(text: str) -> str:
    fenced = re.search(r"```(?:csv)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _as_int(v: object) -> int:
    v2 = str(v).strip()
    if v2 == "":
        return 0
    return int(round(float(v2)))


def _as_float(v: object) -> float:
    v2 = str(v).strip()
    if v2 == "":
        return 0.0
    return float(v2)


def _parse_score_csv(csv_text: str) -> List[ScoreRow]:
    csv_text = _extract_csv_block(csv_text)
    rows: List[ScoreRow] = []

    reader = csv.DictReader(csv_text.splitlines())
    expected = set(OUTPUT_COLUMNS)
    if not reader.fieldnames:
        raise ValueError("Model returned empty CSV.")
    missing = expected.difference(set(reader.fieldnames))
    if missing:
        raise ValueError(f"Model CSV missing columns: {sorted(missing)}")

    for r in reader:
        rows.append(
            ScoreRow(
                story_number=_as_int(r["Story number"]),
                readability=_as_float(r["Readability"]),
                logic=_as_float(r["Logic"]),
                grammar_and_linguistics=_as_float(r["GrammarandLinguistics"]),
                reading_level=str(r["ReadingLevel"]).strip(),
                total_modal_collapse=_as_int(r["TotalModalCollapse"]),
                structure=_as_int(r["Structure"]),
                vocabulary_level=_as_int(r["VocabularyLevel"]),
                stereotypes=_as_int(r["Stereotypes"]),
                gender_balanced=_as_int(r["Gender-balanced"]),
            )
        )
    return rows


def _extract_json_block(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _parse_score_json(json_text: str) -> List[ScoreRow]:
    blob = _extract_json_block(json_text)
    data = json.loads(blob)

    if isinstance(data, dict):
        items = data.get("items")
        rows_data = items if isinstance(items, list) else [data]
    elif isinstance(data, list):
        rows_data = data
    else:
        raise ValueError("JSON response must be an object or list")

    rows: List[ScoreRow] = []
    key_aliases = {
        "storynumber": ["storynumber", "story_no", "storyid"],
        "readability": ["readability"],
        "logic": ["logic"],
        "grammarandlinguistics": ["grammarandlinguistics", "grammarandlinguistic", "grammarlinguistics"],
        "readinglevel": ["readinglevel"],
        "totalmodalcollapse": ["totalmodalcollapse"],
        "structure": ["structure"],
        "vocabularylevel": ["vocabularylevel"],
        "stereotypes": ["stereotypes"],
        "genderbalanced": ["genderbalanced", "genderbalance"],
    }

    for item in rows_data:
        if not isinstance(item, dict):
            raise ValueError("Each JSON row must be an object")

        norm_to_value = {_norm_key(str(k)): v for k, v in item.items()}

        def pick(field: str) -> object:
            for alias in key_aliases[field]:
                if alias in norm_to_value:
                    return norm_to_value[alias]
            raise KeyError(field)

        rows.append(
            ScoreRow(
                story_number=_as_int(pick("storynumber")),
                readability=_as_float(pick("readability")),
                logic=_as_float(pick("logic")),
                grammar_and_linguistics=_as_float(pick("grammarandlinguistics")),
                reading_level=str(pick("readinglevel")).strip(),
                total_modal_collapse=_as_int(pick("totalmodalcollapse")),
                structure=_as_int(pick("structure")),
                vocabulary_level=_as_int(pick("vocabularylevel")),
                stereotypes=_as_int(pick("stereotypes")),
                gender_balanced=_as_int(pick("genderbalanced")),
            )
        )

    return rows


def _looks_like_score_csv(text: str) -> bool:
    block = _extract_csv_block(text)
    return "Story number" in block and "Readability" in block


def _parse_score_response(text: str) -> List[ScoreRow]:
    if _looks_like_score_csv(text):
        try:
            return _parse_score_csv(text)
        except Exception as csv_exc:
            if "```json" in text.lower() or text.strip().startswith(("{", "[")):
                return _parse_score_json(text)
            raise csv_exc
    return _parse_score_json(text)


def score_with_gpt52(stories: Sequence[str], source_path: Path) -> List[ScoreRow]:
    if not stories:
        return []

    client, deployment = create_azure_client()
    batch_size = int(os.getenv("STORY_BATCH_SIZE", "10"))
    all_rows: List[ScoreRow] = []

    for start in range(0, len(stories), batch_size):
        batch = stories[start : start + batch_size]

        buf = io.StringIO()
        w = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
        for s in batch:
            w.writerow([s])
        stories_csv = buf.getvalue().strip()

        retry_note = ""
        batch_rows: List[ScoreRow] = []

        for _attempt in range(3):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous evaluator of Arabic children's stories. "
                        "Output ONLY valid CSV with exactly one row per input story. "
                        "Do not include commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{PROMPT}\n\n"
                        f"Important: The attached CSV contains {len(batch)} stories. "
                        f"Return exactly {len(batch)} rows, with Story number from 1..{len(batch)} in order.\n\n"
                        f"CSV:\n{stories_csv}"
                        + (f"\n\nPrevious output issue to fix: {retry_note}" if retry_note else "")
                    ),
                },
            ]

            content = chat_with_retry(client, model=deployment, messages=messages)
            try:
                batch_rows = _parse_score_response(content)
            except Exception as exc:
                retry_note = str(exc)
                continue

            if len(batch_rows) != len(batch):
                retry_note = f"row_count_mismatch expected={len(batch)} got={len(batch_rows)}"
                batch_rows = []
                continue

            break

        if not batch_rows:
            raise ValueError(
                f"Failed to parse scoring output for {source_path.name} "
                f"batch {start}-{start + len(batch) - 1}"
            )

        for i, row in enumerate(batch_rows, start=1):
            global_idx = start + i
            all_rows.append(
                ScoreRow(
                    story_number=global_idx,
                    readability=row.readability,
                    logic=row.logic,
                    grammar_and_linguistics=row.grammar_and_linguistics,
                    reading_level=row.reading_level,
                    total_modal_collapse=row.total_modal_collapse,
                    structure=row.structure,
                    vocabulary_level=row.vocabulary_level,
                    stereotypes=row.stereotypes,
                    gender_balanced=row.gender_balanced,
                )
            )

    if len(all_rows) != len(stories):
        raise ValueError(
            f"Final row count mismatch for {source_path.name}: "
            f"got {len(all_rows)} for {len(stories)}"
        )

    return all_rows


def write_scores(output_path: Path, rows: Sequence[ScoreRow]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow(row.to_csv_row())


def _existing_score_file_is_complete(score_path: Path, expected_story_count: int) -> bool:
    if not score_path.exists():
        return False
    try:
        with score_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False
            if list(reader.fieldnames) != OUTPUT_COLUMNS:
                return False
            data_rows = list(reader)
        if len(data_rows) != expected_story_count:
            return False
        if data_rows:
            first = data_rows[0]
            if _as_float(first.get("Readability")) is None and str(first.get("Readability", "")).strip() == "":
                return False
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-score story CSVs with Azure OpenAI")
    ap.add_argument("--results-dir", type=Path, default=None, help="Experiment results root")
    ap.add_argument(
        "--input-dirs",
        action="append",
        default=None,
        help="Subfolders under results-dir containing story CSVs (repeatable)",
    )
    args = ap.parse_args()

    egra_results = resolve_results_dir(args.results_dir)
    scores_dir = egra_results / "SCORES"
    scores_dir.mkdir(parents=True, exist_ok=True)

    input_folders = args.input_dirs if args.input_dirs else DEFAULT_INPUT_FOLDERS
    inputs = list(iter_input_csv_files(egra_results, input_folders))
    if not inputs:
        print("No input CSVs found.")
        return 0

    root = repo_root()
    for input_path in inputs:
        stories = read_stories(input_path)
        output_path = scores_dir / f"{input_path.stem}_SCORE.csv"

        if _existing_score_file_is_complete(output_path, expected_story_count=len(stories)):
            print(f"Skipping (already scored): {output_path.name}")
            continue

        print(f"Scoring {input_path.relative_to(root)} ({len(stories)} stories)...")
        rows = score_with_gpt52(stories, input_path)
        write_scores(output_path, rows)
        print(f"Wrote: {output_path.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
