"""Grade-school maths (GSM8K) as a task with one correct answer per problem.

A search over decode paths is judged here by whether any path, and the path it
picks, reaches the right number -- not by how varied the paths are. GSM8K
(Cobbe et al., 2021) is the standard benchmark for search and self-consistency
methods on language models: each problem's reference solution ends with
"#### <number>", and the model is asked to end the same way.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

HF_DATASET = ("openai/gsm8k", "main")

SYSTEM = "You are a careful assistant who solves maths word problems."

INSTRUCTION = (
    "{question}\n\n"
    "Solve the problem step by step. At the end, write the final answer as a single "
    "number on its own line, in the form:\n#### <number>"
)

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
_FINAL = re.compile(r"####\s*(" + _NUM + ")")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANY = re.compile(_NUM)


def _to_float(s: str) -> Optional[float]:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gold_answer(solution: str) -> Optional[float]:
    """The number after '####' in a GSM8K reference solution."""
    m = _FINAL.search(solution)
    return _to_float(m.group(1)) if m else None


def extract_answer(text: str) -> Optional[float]:
    """The model's final answer: the number after the last '####', else the last
    \\boxed{} (instruct models often answer that way), else the last number."""
    finals = _FINAL.findall(text or "")
    if finals:
        return _to_float(finals[-1])
    boxed = _BOXED.findall(text or "")
    if boxed:
        inner = _ANY.findall(boxed[-1])
        if inner:
            return _to_float(inner[-1])
    nums = _ANY.findall(text or "")
    return _to_float(nums[-1]) if nums else None


def is_correct(text: str, gold: float) -> bool:
    pred = extract_answer(text)
    return pred is not None and gold is not None and abs(pred - gold) < 1e-6


def load_problems(n: int, *, seed: int = 0, split: str = "test") -> List[Dict]:
    """``n`` problems drawn at random (fixed seed) from GSM8K's split, with gold answers."""
    from datasets import load_dataset

    ds = load_dataset(*HF_DATASET, split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        row = ds[i]
        out.append({"index": i, "question": row["question"].strip(),
                    "answer": gold_answer(row["answer"])})
    return out


def build_messages(question: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION.format(question=question.strip())}]
