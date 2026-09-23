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
    "baseline", "sampling", "steer", "per-token", "per-story", "f(Sc)", "amplify",
    "embed", "attn", "resid", "entropy", "double", "two-stage", "prior", "other",
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
    "dia": "dialogue", "len": "length", "ter": "short sentences",
    "var": "varied sentence openings", "sho": "short words",
}

# How f(S_c) perturbs the constraint vector. See noiseegra.subspace.
JITTER_MODES = {
    "perp": "sideways step",
    "rotate": "turned",
    "gain": "constraint mix jittered",
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
_GATE = re.compile(r"__gate([a-z]+)")
_NHORIZON = re.compile(r"__nh(\d+)")
_NSCHED = re.compile(r"__nsch([a-z])")
_AMP = re.compile(rf"__amp(?P<val>m?{_FLOAT})")
_JIT = re.compile(rf"__j(?P<val>m?{_FLOAT})(?P<mode>perp|rotate|gain)")
_BUDGET = re.compile(rf"__bud(?P<val>m?{_FLOAT})")
# The per-constraint steering schedule, as codes: c constant, d cosine decay,
# r ramp, l linear decay, p prefix. Not to be confused with __nsch, which is the
# *noise* schedule.
_SSCHED = re.compile(r"__sch(?!eme)([a-z](?:-[a-z])*)(?![a-z])")
_SSCHED_NAME = {"d": "decaying over the opening", "r": "ramping up over the story",
                "l": "falling away over the story", "p": "only over the opening"}


def _steer_schedule(run_id: str) -> str:
    """How the constraint push is shaped over the story, when it is not flat."""
    m = _SSCHED.search(run_id)
    if not m:
        return ""
    codes = set(m.group(1).split("-"))
    if len(codes) != 1:
        # Mixed per-constraint schedules have no short name, and the older runs
        # that used them (closure ramping while the rest stayed flat) already have
        # their schedules described in the table header.
        return ""
    code = codes.pop()
    return "" if code == "c" else ", " + _SSCHED_NAME.get(code, f"schedule {code}")
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
    short: str = ""

    def __post_init__(self):
        if not self.short:
            self.short = _shorten(self.text)

    @property
    def sort_key(self):
        order = FAMILY_ORDER.index(self.family) if self.family in FAMILY_ORDER else 99
        return (order, self.magnitude, self.text)


_SHORTEN = [
    ("steering only, no perturbation", "steer"),
    ("per-token noise a=", "tok "),
    ("per-story offset g=", "sto "),
    (", from the prompt onward", "+pre"),
    (", at the prompt only", " pre"),
    ("story-difference amplified x", "amp "),
    (", only over the first ", " <"),
    (" (orthogonal)", ""), (" (unrestricted)", " iso"), (" (in-subspace)", " para"),
    (", gated to the most uncertain steps", " g-hi"),
    (", gated to uncertain steps", " g-med"),
    ("residual-stream noise", "resid"), ("attention-logit noise", "attn"),
    ("embedding noise", "embed"), ("AENI entropy-scaled noise", "aeni"),
    ("residual noise at two sites", "resid2"),
    ("two-stage plan then write, residual noise", "2stage+n"),
    ("two-stage plan then write, no noise", "2stage"),
    ("baseline (temperature ", "T"), ("baseline", "base"),
    (", top-k ", "/k"), (", top-p ", "/p"),
]


def _shorten(text: str) -> str:
    """A few characters for a column header in a transposed table."""
    out = text
    for long, short in _SHORTEN:
        out = out.replace(long, short)
    out = re.sub(r"\s*\([^)]*\)", "", out).strip(" ,;()")
    return re.sub(r"\s+", " ", out)[:14].strip(" ,;()")


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


def _noise_window(run_id: str) -> str:
    """How the perturbation is spread over the story, if it is not flat."""
    sc = _NSCHED.search(run_id)
    if not sc:
        return ""
    h = _NHORIZON.search(run_id)
    steps = f"{h.group(1)} tokens" if h else "the horizon"
    return {
        "p": f", only over the first {steps}",
        "d": f", fading out over the first {steps}",
        "l": f", fading out linearly over the first {steps}",
        "r": f", ramping up over the first {steps}",
    }.get(sc.group(1), "")


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

    if "__PRIOR__" in rid:
        from .prior_methods import NAMES
        method = rid.split("__PRIOR__", 1)[1].split("__", 1)[0]
        return RunLabel(NAMES.get(method, method), "prior", 0.0, rid)

    if "__ORTHO" in up:
        # Where the constraint vector itself is applied. Every f(S_c) arm and every
        # steering-only control comes in both sitings, so the table has to say
        # which one a row is.
        site = " at the prompt" if "__sdec0" in rid else ""
        if "__dirrandom" in rid:
            site += ", random directions"
        gate = _GATE.search(rid)
        gate_txt = ({"median": ", gated to uncertain steps",
                     "high": ", gated to the most uncertain steps"}
                    .get(gate.group(1), f", gate {gate.group(1)}") if gate else "")
        amp = _AMP.search(rid)
        if amp:
            v = untag_float(amp.group("val"))
            where = " from the prompt" if "__apre" in rid else " while writing"
            return RunLabel(
                f"story-difference amplified x{_fmt(v)}{where}{gate_txt}",
                "amplify", v, rid)
        # Whether a constraint push rides under the perturbation, and how it is
        # shaped. Said in the row because one run can hold the same perturbation
        # over different pushes (with and without, fewer directions, a schedule)
        # and those must not share a name. Only budgeted pushes are named: the
        # pre-budget runs steered under every perturbed arm, so their labels stay
        # as the older tables printed them.
        bud = _BUDGET.search(rid)
        push_txt = ""
        if bud:
            push_txt = f" + push at {_fmt(untag_float(bud.group('val')))}"
            m = _BETAS.search(rid)
            if m:
                betas_ = m.group(1).split("-")
                n_on = sum(1 for t in betas_ if untag_float(t) != 0)
                if 0 < n_on < len(betas_):
                    push_txt += f" on {n_on} directions"
            push_txt += _steer_schedule(rid)

        g = _G.search(rid)
        if g and g.group("mode") != "none" and untag_float(g.group("val")) > 0:
            v = untag_float(g.group("val"))
            mode = SUBSPACE_MODES.get(g.group("mode"), g.group("mode"))
            # An offset with no estimated basis is a random direction of the same
            # length -- the control for whether the story-difference basis, and
            # not the mere fact of a per-story shift, is where the diversity
            # comes from. Said in the row because a run can mix the two.
            if "__obiso" in rid:
                mode += ", random direction"
            if "__ponly" in rid:
                where = ", at the prompt only"
            elif "__opre" in rid:
                where = ", from the prompt onward"
            else:
                where = ""
            if "__odspread" in rid:
                where += ", the set chosen together"
            if "__online" in rid:
                where += ", sized while writing"
            m = re.search(r"__otilt(\d+p?\d*)", rid)
            if m:
                where += f", output tilted toward the rules at {_fmt(untag_float(m.group(1)))}"
            return RunLabel(
                f"per-story offset g={_fmt(v)} ({mode}){where}{push_txt}{site}{gate_txt}",
                "per-story", v, rid)
        a = _NZ.search(rid)
        if a and a.group("mode") != "none" and untag_float(a.group("val")) > 0:
            v = untag_float(a.group("val"))
            mode = SUBSPACE_MODES.get(a.group("mode"), a.group("mode"))
            return RunLabel(f"per-token noise a={_fmt(v)} ({mode})"
                            f"{_noise_window(rid)}{push_txt}{site}{gate_txt}",
                            "per-token", v, rid)
        bud_txt = ""
        if bud:
            n_on = 0
            m = _BETAS.search(rid)
            if m:
                n_on = sum(1 for t in m.group(1).split("-") if untag_float(t) != 0)
            bud_txt = (f", {n_on} direction{'s' if n_on != 1 else ''}, total push held "
                       f"at {_fmt(untag_float(bud.group('val')))}")
        j = _JIT.search(rid)
        if j:
            v = untag_float(j.group("val"))
            mode = j.group("mode")
            draw = ", drawn in the activation subspace" if "__jdbasis" in rid else ""
            if mode == "gain":
                text = (f"the constraint budget reallocated per story, spread {_fmt(v)}"
                        if bud else f"constraint mix jittered, spread {_fmt(v)}")
            elif mode == "rotate":
                text = f"f(S_c): constraint vector turned, kappa={_fmt(v)}"
            else:
                text = f"f(S_c): sideways step on the constraint vector, kappa={_fmt(v)}"
            if mode == "gain" and bud:
                text += bud_txt
                text += _steer_schedule(rid)
            return RunLabel(f"{text}{bud_txt if mode != 'gain' else ''}{draw}{site}{gate_txt}",
                            "f(Sc)", v, rid)
        return RunLabel(
            f"steering only, no perturbation{bud_txt}{_steer_schedule(rid)}{site}",
            "steer", 0.0, rid)

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

    gates = {m.group(1) for m in map(_GATE.search, run_ids) if m}
    if not gates and all("__gate" not in r for r in run_ids):
        out.append("perturbation is applied at every decode step (no entropy gate)")

    def _basis_kind(rid: str) -> str:
        for tag in ("story", "prompt", "iso"):
            if f"__ob{tag}" in rid:
                return tag
        return "step"

    kinds = {_basis_kind(r) for r in run_ids if _G.search(r) and "__g0orth" not in r}
    if len(kinds) == 1:
        out.append(
            "per-story offsets are drawn from the directions along which "
            + {"story": "whole stories differ from one another",
               "prompt": "the instruction's own token positions differ from one another",
               "iso": "nothing in particular: an isotropic random draw, the "
                      "control for the estimated basis",
               "step": "one decode step differs from another"}[kinds.pop()]
        )

    windows = {(m.group(1), _NHORIZON.search(r).group(1) if _NHORIZON.search(r) else "")
               for r in run_ids for m in [_NSCHED.search(r)] if m}
    if len(windows) == 1:
        sched, h = windows.pop()
        if sched == "p" and h:
            out.append(f"the perturbation covers only the first {h} generated tokens")

    if any(_JIT.search(r) for r in run_ids):
        out.append(
            "f(S_c) rows perturb the constraint vector itself rather than adding a "
            "perturbation beside it, so their perturbation is free to lie inside the "
            "constraint subspace; what is held fixed is the push along the summed "
            "constraint vector"
        )

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
        out.append("schedules: the steering strength is flat over the story")
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
