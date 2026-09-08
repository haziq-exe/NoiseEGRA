"""Plot-level diversity: reduce each story to a skeleton, then measure diversity there.

Following Echoes in AI (Xu et al., 2025), which measures plot diversity rather than
surface diversity. The motivation for us is specific: our constraints are all
surface-level (word count, tense, reading level, dialogue), and perturbation moved
those. Diversity over the prose therefore risks rewarding style churn or outright
degradation. Diversity over a plot skeleton asks whether *different things happen*.

Extraction is mechanical, not evaluative. The model is asked to fill six fixed
slots -- it never scores or ranks anything, so this is not an LLM judge and the
diversity computation downstream stays fully automatic.

Two backends:

``hf``     an instruct model fills the slots. Better skeletons, needs a GPU.
``spacy``  characters, places and main verbs pulled out with POS tags and NER.
           No GPU, no generation, much coarser.

Extractions are cached to JSON keyed by a hash of the story text, so re-scoring
the same runs costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PLOT_FIELDS = ("setting", "protagonist", "goal", "obstacle", "turning_point", "resolution")

PLOT_SYSTEM = (
    "You extract structured plot summaries. You reply with JSON only, no commentary."
)

PLOT_PROMPT = """Read the story and fill in this plot skeleton.

Story:
\"\"\"{story}\"\"\"

Reply with a JSON object with exactly these keys, each a short phrase of at most
eight words. Describe what actually happens; do not evaluate the story.

{{"setting": "", "protagonist": "", "goal": "", "obstacle": "", "turning_point": "", "resolution": ""}}

If the story does not supply one of these, write "none" for that key."""

_JSON = re.compile(r"\{.*?\}", re.S)
_KEYVAL = re.compile(r'"?([a-z_]+)"?\s*[:=]\s*"?([^"\n,}]+)"?')


def story_key(text: str, tag: str = "") -> str:
    return hashlib.sha1(f"{tag}\x00{text}".encode("utf-8")).hexdigest()


def parse_skeleton(raw: str) -> Dict[str, str]:
    """Pull the six slots out of a model reply, tolerating malformed JSON."""
    out = {f: "none" for f in PLOT_FIELDS}
    match = _JSON.search(raw or "")
    if match:
        try:
            data = json.loads(match.group(0))
            for f in PLOT_FIELDS:
                v = data.get(f)
                if isinstance(v, str) and v.strip():
                    out[f] = v.strip()
            return out
        except json.JSONDecodeError:
            pass
    for key, val in _KEYVAL.findall(raw or ""):
        if key in out and val.strip():
            out[key] = val.strip()
    return out


def skeleton_text(skel: Dict[str, str]) -> str:
    """Canonical string for embedding. Fixed field order, so the shape is constant
    across stories and only the content varies."""
    return " ".join(f"{f}: {skel.get(f, 'none')}." for f in PLOT_FIELDS)


class PlotCache:
    def __init__(self, path: Optional[Path | str]):
        self.path = Path(path) if path else None
        self.data: Dict[str, Dict[str, str]] = {}
        if self.path and self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.data = {}

    def get(self, key: str) -> Optional[Dict[str, str]]:
        return self.data.get(key)

    def put(self, key: str, skel: Dict[str, str]) -> None:
        self.data[key] = skel

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


class HFPlotExtractor:
    """Fills the plot slots with a local instruct model, greedily."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        dtype: str = "auto",
        max_new_tokens: int = 160,
        batch_size: int = 8,
        max_story_words: int = 400,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.max_story_words = max_story_words
        self._model = None
        self._tok = None
        self._dtype = dtype

    @property
    def tag(self) -> str:
        return f"hf:{self.model_id}"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._dtype == "auto":
            cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
            torch_dtype = torch.bfloat16 if cap >= 8 else torch.float16
        else:
            torch_dtype = getattr(torch, self._dtype)
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        self._model.eval()

    def extract(self, stories: Sequence[str]) -> List[Dict[str, str]]:
        import torch

        self._load()
        out: List[Dict[str, str]] = []
        for i in range(0, len(stories), self.batch_size):
            chunk = stories[i : i + self.batch_size]
            texts = []
            for s in chunk:
                clipped = " ".join(s.split()[: self.max_story_words])
                msgs = [
                    {"role": "system", "content": PLOT_SYSTEM},
                    {"role": "user", "content": PLOT_PROMPT.format(story=clipped)},
                ]
                texts.append(
                    self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                )
            enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=2048).to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tok.pad_token_id,
                )
            for row, prompt_len in zip(gen, [enc["input_ids"].shape[1]] * len(chunk)):
                reply = self._tok.decode(row[prompt_len:], skip_special_tokens=True)
                out.append(parse_skeleton(reply))
        return out


class SpacyPlotExtractor:
    """Rule-based skeleton: who, where, and what they do. No generation.

    Coarser than the model backend -- it captures the cast and the actions but not
    goal/obstacle/resolution structure -- but it needs no GPU and is fully
    deterministic, which makes it a useful check that a plot-level result is not an
    artefact of the extraction model.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", max_items: int = 8):
        self.spacy_model = spacy_model
        self.max_items = max_items
        self._nlp = None

    @property
    def tag(self) -> str:
        return f"spacy:{self.spacy_model}"

    def _load(self):
        if self._nlp is not None:
            return
        import spacy

        self._nlp = spacy.load(self.spacy_model, disable=["lemmatizer"])

    def extract(self, stories: Sequence[str]) -> List[Dict[str, str]]:
        self._load()
        out = []
        for doc in self._nlp.pipe(list(stories), batch_size=32):
            people = _uniq([e.text for e in doc.ents if e.label_ == "PERSON"], self.max_items)
            places = _uniq(
                [e.text for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC", "ORG")],
                self.max_items,
            )
            verbs = _uniq(
                [t.lemma_.lower() for t in doc if t.pos_ == "VERB" and not t.is_stop],
                self.max_items,
            )
            objects = _uniq(
                [t.lemma_.lower() for t in doc if t.dep_ in ("dobj", "pobj") and t.pos_ == "NOUN"],
                self.max_items,
            )
            out.append({
                "setting": ", ".join(places) or "none",
                "protagonist": ", ".join(people) or "none",
                "goal": "none",
                "obstacle": "none",
                "turning_point": ", ".join(verbs) or "none",
                "resolution": ", ".join(objects) or "none",
            })
        return out


def _uniq(items: Sequence[str], limit: int) -> List[str]:
    seen, out = set(), []
    for it in items:
        k = it.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
        if len(out) >= limit:
            break
    return out


def extract_skeletons(
    stories: Sequence[str], extractor, cache: Optional[PlotCache] = None
) -> List[Dict[str, str]]:
    """Skeletons for ``stories``, reading and filling ``cache`` as it goes."""
    tag = getattr(extractor, "tag", extractor.__class__.__name__)
    keys = [story_key(s, tag) for s in stories]
    result: List[Optional[Dict[str, str]]] = [
        cache.get(k) if cache is not None else None for k in keys
    ]
    # Dedupe within the call as well as against the cache: modal collapse makes
    # identical stories common, and extraction is the expensive step.
    todo: Dict[str, int] = {}
    for i, r in enumerate(result):
        if r is None:
            todo.setdefault(keys[i], i)
    if todo:
        order = list(todo.values())
        fresh = extractor.extract([stories[i] for i in order])
        by_key = {keys[i]: skel for i, skel in zip(order, fresh)}
        for i, k in enumerate(keys):
            if result[i] is None and k in by_key:
                result[i] = by_key[k]
        if cache is not None:
            for k, skel in by_key.items():
                cache.put(k, skel)
            cache.save()
    return [r or {f: "none" for f in PLOT_FIELDS} for r in result]
