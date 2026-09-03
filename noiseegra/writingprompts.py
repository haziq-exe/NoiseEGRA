"""WritingPrompts task setup: prompt selection and constrained-prompt construction.

Prompts come from the WritingPrompts corpus (Fan et al., 2018), scraped from
r/WritingPrompts -- the same source the CS4 creativity benchmark builds on. We
pair it with a *fixed* set of verifiable constraints rather than CS4's
per-instance synthesized ones, because a steering direction has to mean the same
thing on every prompt: there is no single direction for "set in a coastal town",
but there is one for "present tense".

Diversity has to be measured **within** a prompt. Stories written from different
prompts are trivially dissimilar, so a diversity score pooled across prompts would
mostly measure the prompt set. The runner therefore generates K stories per prompt
and the scorer computes diversity per prompt group before averaging.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HF_DATASET = "euclaise/writingprompts"

# r/WritingPrompts posts carry a bracketed tag, e.g. "[ WP ] You wake up ...".
_TAG = re.compile(r"^\s*[\[\(]\s*(WP|EU|CW|TT|RF|IP|MP|PM|PI|SP|OT)\s*[\]\)]\s*", re.I)
_WS = re.compile(r"\s+")

# A steering vector and the constraint it targets do not always share a name:
# the direction is "closure" (wrap the story up) but the constraint it moves is
# "length" (word count). The pair file records this in each entry's `targets`.
VECTOR_TO_CONSTRAINT = {
    "closure": "length",
    "present_tense": "present_tense",
    "simple_register": "simple_register",
    "dialogue": "dialogue",
}


def as_constraint(name: str) -> str:
    """Map a steering-vector name to the constraint it targets (identity if already one)."""
    return VECTOR_TO_CONSTRAINT.get(name, name)


CONSTRAINT_TEXT = {
    "length": "The story must be at most {max_words} words long.",
    "present_tense": "The story must be written entirely in the present tense.",
    "simple_register": (
        "The story must use plain, accessible language, at roughly a "
        "{max_grade:.0f}th-grade reading level: short sentences and common words."
    ),
    "dialogue": "The story must include at least one line of spoken dialogue in quotation marks.",
}

SYSTEM_PROMPT = "You are a creative writer."

INSTRUCTION = (
    "{prompt}\n\n"
    "Write a short story in response. It must follow every one of these requirements:\n"
    "{constraints}\n\n"
    "Write only the story itself, with no title, heading, preamble, or commentary."
)

# Used as the context for steering-vector extraction. Deliberately generic and
# disjoint from the evaluation prompts, so the directions are not fitted to them.
EXTRACTION_PROMPT = (
    "Write a short story.\n\n"
    "Write only the story itself, with no title, heading, preamble, or commentary."
)


def clean_prompt(text: str) -> str:
    return _WS.sub(" ", _TAG.sub("", text or "")).strip()


def load_prompts(
    n: int,
    *,
    seed: int = 0,
    split: str = "test",
    min_words: int = 8,
    max_words: int = 60,
    cache: Optional[str | Path] = None,
) -> List[str]:
    """Select ``n`` prompts deterministically from WritingPrompts.

    Writes the selection to ``cache`` (JSON) and reuses it on later calls, so a
    resumed run uses exactly the same prompts without re-downloading.
    """
    if cache is not None:
        p = Path(cache)
        if p.is_file():
            saved = json.loads(p.read_text(encoding="utf-8"))
            if saved.get("n") == n and saved.get("seed") == seed and saved.get("split") == split:
                return saved["prompts"]

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the `datasets` package is required to load WritingPrompts: pip install datasets"
        ) from exc

    ds = load_dataset(HF_DATASET, split=split)
    field = next(
        (f for f in ("prompt", "prompts", "instruction", "text") if f in ds.column_names),
        None,
    )
    if field is None:
        raise KeyError(
            f"cannot find a prompt column in {HF_DATASET} (columns: {ds.column_names})"
        )

    seen, pool = set(), []
    for raw in ds[field]:
        text = clean_prompt(raw)
        wc = len(text.split())
        if not (min_words <= wc <= max_words):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        pool.append(text)

    if len(pool) < n:
        raise ValueError(f"only {len(pool)} prompts passed the length filter; asked for {n}")

    rng = random.Random(seed)
    prompts = rng.sample(pool, n)

    if cache is not None:
        p = Path(cache)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"dataset": HF_DATASET, "split": split, "seed": seed, "n": n,
                 "pool_size": len(pool), "prompts": prompts},
                indent=2,
            ),
            encoding="utf-8",
        )
    return prompts


def constraint_block(
    constraints: Sequence[str],
    *,
    max_words: int = 150,
    max_grade: float = 6.0,
) -> str:
    lines, seen = [], set()
    for name in constraints:
        key = as_constraint(name)
        if key not in CONSTRAINT_TEXT:
            raise KeyError(
                f"no prompt text for '{name}' (resolved to '{key}'); "
                f"known: {sorted(CONSTRAINT_TEXT)}"
            )
        if key in seen:
            continue
        seen.add(key)
        lines.append("- " + CONSTRAINT_TEXT[key].format(max_words=max_words, max_grade=max_grade))
    return "\n".join(lines)


def build_messages(
    prompt: str,
    constraints: Sequence[str],
    *,
    max_words: int = 150,
    max_grade: float = 6.0,
    system: str = SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": INSTRUCTION.format(
                prompt=prompt.strip(),
                constraints=constraint_block(
                    constraints, max_words=max_words, max_grade=max_grade
                ),
            ),
        },
    ]
