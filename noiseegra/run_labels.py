"""Turn a run id back into something a person can read.

Run ids encode the whole experiment plan so that two conditions can never collide
on disk:

    Granite-3.1-8B__ORTHO__L17-25__Cclo-pre-sim-dia__b1-1-1-1__lowdin__nzorth__a0p4__k36
    ALLam__ATTN__L12-20__std0p105__maxtok200

That is the right property for a filename and the wrong one for a results table.
``label_run`` reads the id back and returns a short name ("per-token noise
a=0.4"), the family it belongs to, and a magnitude, so tables can be sorted by
what actually varies instead of alphabetically by hyperparameter string.

``plan_summary`` pulls out the settings that are the *same* across a sweep --
model, layer band, constraints, orthogonalisation -- so they can be printed once
in a header rather than repeated in every row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Order families appear in a table: the reference conditions first, then the
# thing being swept.
FAMILY_ORDER = [
    "baseline", "sampling", "steer", "per-token", "per-story",
    "embed", "attn", "resid", "entropy", "double", "two-stage", "other",
]

# What the perturbation is allowed to move along, relative to the constraint
# directions the steering uses.
SUBSPACE_MODES = {
    "orth": "orthogonal",      # kept off the constraint directions
    "iso": "unrestricted",     # free to move anywhere
    "para": "in-subspace",     # confined to the constraint directions
    "free": "unrestricted",
}

CONSTRAINT_NAMES = {
    "clo": "closure", "pre": "present tense", "sim": "simple register",
    "dia": "dialogue", "len": "length",
}

_FLOAT = r"[0-9]+(?:p[0-9]+)?"
_LAYERS = re.compile(r"__L(\d+)-(\d+)")
_NZ = re.compile(rf"__nz(?P<mode>[a-z]+)__a(?P<val>m?{_FLOAT})")
_G = re.compile(rf"__g(?P<val>m?{_FLOAT})(?P<mode>[a-z]+)")
_STD = re.compile(rf"__std(?P<val>m?{_FLOAT})")
_STD12 = re.compile(rf"__std1(?P<a>m?{_FLOAT})__std2(?P<b>m?{_FLOAT})")
_TEMP = re.compile(rf"temp(?P<val>m?{_FLOAT})")
_TOPK = re.compile(r"topk(?P<val>\d+)")
_TOPP = re.compile(rf"topp(?P<val>m?{_FLOAT})")
_BETAS = re.compile(r"__b([0-9pm-]+)__")
_CONSTR = re.compile(r"__C([a-z-]+)__")
_PROTECT = re.compile(r"__k(\d+)")
_ORTHONORM = re.compile(r"__(lowdin|gram_schmidt|none)__")


def untag_float(tag: str) -> float:
    """Inverse of the run id float encoding: '0p4' -> 0.4, 'm1' -> -1.0."""
    try:
        return float(tag.replace("p", ".").replace("m", "-"))
    except (ValueError, AttributeError):
        return 0.0


def _fmt(x: float) -> str:
    return f"{x:g}"


@dataclass
class RunLabel:
    text: str
    family: str
    magnitude: float = 0.0
    run_id: str = ""

    @property
    def sort_key(self):
        order = FAMILY_ORDER.index(self.family) if self.family in FAMILY_ORDER else 99
        return (order, self.magnitude, self.text)


def _sampling_suffix(run_id: str) -> str:
    bits = []
    t = _TEMP.search(run_id)
    if t:
        bits.append(f"temperature {_fmt(untag_float(t.group('val')))}")
    k = _TOPK.search(run_id)
    if k:
        bits.append(f"top-k {k.group('val')}")
    p = _TOPP.search(run_id)
    if p:
        bits.append(f"top-p {_fmt(untag_float(p.group('val')))}")
    return ", ".join(bits)


def _layer_suffix(run_id: str) -> str:
    m = _LAYERS.search(run_id)
    return f"layers {m.group(1)}-{m.group(2)}" if m else ""


def _join(name: str, *parts: str) -> str:
    extra = [p for p in parts if p]
    return f"{name} ({'; '.join(extra)})" if extra else name


def label_run(run_id: str) -> RunLabel:
    """Readable name, family and magnitude for one run id."""
    rid = run_id
    up = rid.upper()

    if "__ORTHO" in up:
        g = _G.search(rid)
        if g and g.group("mode") != "none" and untag_float(g.group("val")) > 0:
            v = untag_float(g.group("val"))
            mode = SUBSPACE_MODES.get(g.group("mode"), g.group("mode"))
            return RunLabel(f"per-story offset g={_fmt(v)} ({mode})",
                            "per-story", v, rid)
        a = _NZ.search(rid)
        if a and a.group("mode") != "none" and untag_float(a.group("val")) > 0:
            v = untag_float(a.group("val"))
            mode = SUBSPACE_MODES.get(a.group("mode"), a.group("mode"))
            return RunLabel(f"per-token noise a={_fmt(v)} ({mode})",
                            "per-token", v, rid)
        return RunLabel("steering only, no perturbation", "steer", 0.0, rid)

    if "BASELINE" in up:
        sampling = _sampling_suffix(rid)
        if sampling:
            return RunLabel(f"baseline ({sampling})", "sampling", 1.0, rid)
        return RunLabel("baseline", "baseline", 0.0, rid)

    if "DOUBLE_RESID" in up:
        m = _STD12.search(rid)
        stds = (f"sigma {_fmt(untag_float(m.group('a')))} then "
                f"{_fmt(untag_float(m.group('b')))}") if m else ""
        return RunLabel(_join("residual noise at two sites", _layer_suffix(rid), stds),
                        "double", 0.0, rid)

    if "TWOSTAGE_ZERO" in up:
        return RunLabel("two-stage plan then write, no noise", "two-stage", 0.0, rid)

    if "TWOSTAGE_RESID" in up:
        m = _STD.search(rid)
        std = f"sigma {_fmt(untag_float(m.group('val')))}" if m else ""
        return RunLabel(_join("two-stage plan then write, residual noise",
                              _layer_suffix(rid), std), "two-stage", 1.0, rid)

    if "ENTROPY" in up:
        m = _STD.search(rid)
        std = f"sigma {_fmt(untag_float(m.group('val')))}" if m else ""
        return RunLabel(_join("AENI entropy-scaled noise", _layer_suffix(rid), std),
                        "entropy", 0.0, rid)

    if "__ATTN" in up:
        m = _STD.search(rid)
        std = f"sigma {_fmt(untag_float(m.group('val')))}" if m else ""
        cap = "first 200 tokens" if "maxtok200" in rid else ""
        return RunLabel(_join("attention-logit noise", _layer_suffix(rid), std, cap),
                        "attn", 0.0, rid)

    if "__EMBED" in up:
        m = _STD.search(rid)
        std = f"sigma {_fmt(untag_float(m.group('val')))}" if m else ""
        return RunLabel(_join("embedding noise", std), "embed", 0.0, rid)

    if _LAYERS.search(rid) and _STD.search(rid):
        m = _STD.search(rid)
        return RunLabel(_join("residual-stream noise", _layer_suffix(rid),
                              f"sigma {_fmt(untag_float(m.group('val')))}"),
                        "resid", 0.0, rid)

    return RunLabel(rid, "other", 0.0, rid)


def label_of(run_id: str) -> str:
    return label_run(run_id).text


def plan_summary(run_ids: Sequence[str]) -> List[str]:
    """Settings shared by every run, for a header line instead of a per-row column."""
    if not run_ids:
        return []
    rid = run_ids[0]
    out: List[str] = []

    models = {r.split("__")[0] for r in run_ids}
    if len(models) == 1:
        out.append(f"model: {models.pop()}")

    bands = {(m.group(1), m.group(2)) for m in map(_LAYERS.search, run_ids) if m}
    if len(bands) == 1:
        lo, hi = bands.pop()
        out.append(f"steering layers: {lo}-{hi}")

    c = _CONSTR.search(rid)
    if c:
        names = [CONSTRAINT_NAMES.get(p, p) for p in c.group(1).split("-")]
        out.append(f"constraints: {', '.join(names)}")

    b = _BETAS.search(rid)
    if b:
        betas = [_fmt(untag_float(x)) for x in b.group(1).split("-")]
        out.append("steering strength: beta " +
                   (betas[0] if len(set(betas)) == 1 else ", ".join(betas)))

    o = _ORTHONORM.search(rid)
    if o:
        how = {"lowdin": "Lowdin (symmetric, order-independent)",
               "gram_schmidt": "Gram-Schmidt (order-dependent)",
               "none": "none"}[o.group(1)]
        out.append(f"directions made mutually orthogonal by: {how}")

    k = _PROTECT.search(rid)
    if k:
        out.append(f"protected subspace: {k.group(1)} dimensions "
                   "(perturbation is kept out of these)")

    modes = {m.group("mode") for m in map(_NZ.search, run_ids) if m}
    modes |= {m.group("mode") for m in map(_G.search, run_ids) if m}
    modes.discard("none")
    if len(modes) == 1:
        mode = modes.pop()
        meaning = {
            "orth": "orthogonal to the constraint directions, so it cannot push a "
                    "constraint either way",
            "iso": "unrestricted, free to move along any direction",
            "para": "confined to the constraint directions",
        }.get(mode, mode)
        out.append(f"perturbation is {meaning}")

    if all("schr-c-c-c" in r or "__sch" not in r for r in run_ids):
        out.append("schedules: closure ramps up over the story, the others constant")
    if all("__nsch" not in r for r in run_ids):
        out.append("perturbation magnitude is constant over the story (no decay)")
    return out


def sorted_labels(run_ids: Sequence[str]) -> List[RunLabel]:
    """Labels ordered by family then magnitude, with duplicates disambiguated."""
    labels = [label_run(r) for r in run_ids]
    seen: Dict[str, int] = {}
    for lab in labels:
        seen[lab.text] = seen.get(lab.text, 0) + 1
    for lab in labels:
        if seen[lab.text] > 1:
            lab.text = f"{lab.text} [{lab.run_id[-10:]}]"
    return sorted(labels, key=lambda l: l.sort_key)


def label_from_run_id(run_id: str) -> Optional[str]:
    """Back-compatible shim: ``None`` when the id is not recognised."""
    lab = label_run(run_id)
    return None if lab.family == "other" else lab.text
