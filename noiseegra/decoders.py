"""Published truncation schemes, applied as an explicit logits processor.

Locally typical sampling (Meister et al., TACL 2023), eta-sampling (Hewitt et
al., EMNLP Findings 2022) and min-p (Nguyen et al., ICLR 2025) are all settings
the generation stack once accepted as keyword arguments to ``generate``. They
are not reliably read any more: a ``GenerationConfig`` accepts any attribute you
set on it, so a key the installed version no longer looks at is stored, ignored,
and never reported. Six conditions differing only in these three settings came
back byte-identical to plain nucleus sampling across 200 stories each, and
nothing in the log said so.

Passing an explicit processor removes the dependency on which keys that version
happens to read. It also fixes the arithmetic in one place, against the
published definition, rather than inheriting whatever the installed release
does.

Temperature is applied here rather than by the stack's own warper, and the
caller therefore asks ``generate`` for temperature 1.0. Order is the reason: a
custom processor runs before the built-in warpers, so a relative threshold like
min-p would otherwise be measured on unscaled logits and mean something
different at every temperature.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor


class TruncationWarper(LogitsProcessor):
    """One published truncation scheme, plus the temperature it is measured at.

    Exactly one of ``typical_p``, ``min_p`` or ``eta_cutoff`` is honoured; the
    caller is expected to pass one. Whatever the scheme would remove, the single
    most likely token is always kept, so the distribution can never be empty.

    It subclasses ``LogitsProcessor`` because the generation stack checks: a
    plain object with the right ``__call__`` is accepted by transformers 4 and
    quietly dropped by transformers 5, which is how five decoding conditions
    came back byte-identical to plain nucleus sampling for a second time after
    the first cause had been fixed.

    ``calls`` counts how many times the stack actually invoked it, so a run can
    show in its log that the setting arrived rather than assert that it did.
    """

    def __init__(self, *, temperature=1.0, typical_p=None, min_p=None,
                 eta_cutoff=None, filter_value=-float("inf")):
        chosen = [x is not None for x in (typical_p, min_p, eta_cutoff)]
        if sum(chosen) != 1:
            raise ValueError(
                "exactly one of typical_p, min_p, eta_cutoff must be given; "
                f"got typical_p={typical_p}, min_p={min_p}, eta_cutoff={eta_cutoff}")
        self.temperature = float(temperature)
        self.typical_p = None if typical_p is None else float(typical_p)
        self.min_p = None if min_p is None else float(min_p)
        self.eta_cutoff = None if eta_cutoff is None else float(eta_cutoff)
        self.filter_value = filter_value
        self.calls = 0

    # The signature must match the base class exactly. The generation stack
    # inspects it and refuses anything it cannot supply every parameter for, so
    # adding **kwargs here -- which looks like defensive coding -- makes it
    # demand a `kwargs` argument and raise.
    def __call__(self, input_ids, scores):
        self.calls += 1
        if self.temperature != 1.0:
            scores = scores / self.temperature
        if self.min_p is not None:
            remove = self._min_p_mask(scores)
        elif self.eta_cutoff is not None:
            remove = self._eta_mask(scores)
        else:
            remove = self._typical_mask(scores)
        # Never empty the row: the single most likely token always survives.
        top = scores.argmax(dim=-1, keepdim=True)
        remove.scatter_(1, top, False)
        return scores.masked_fill(remove, self.filter_value)

    def _min_p_mask(self, scores):
        """Nguyen et al.: keep tokens with p >= min_p * max_p."""
        probs = torch.softmax(scores, dim=-1)
        top = probs.max(dim=-1, keepdim=True).values
        return probs < (self.min_p * top)

    def _eta_mask(self, scores):
        """Hewitt et al.: keep p >= min(eta, sqrt(eta) * exp(-H)), H in nats."""
        logp = torch.log_softmax(scores, dim=-1)
        probs = logp.exp()
        entropy = -(probs * logp).sum(dim=-1, keepdim=True)
        eps = torch.minimum(
            torch.full_like(entropy, self.eta_cutoff),
            (self.eta_cutoff ** 0.5) * torch.exp(-entropy))
        return probs < eps

    def _typical_mask(self, scores):
        """Meister et al.: the smallest set, by distance of -log p from the
        entropy, whose probability reaches typical_p."""
        logp = torch.log_softmax(scores, dim=-1)
        probs = logp.exp()
        entropy = -(probs * logp).sum(dim=-1, keepdim=True)
        deviation = (-logp - entropy).abs()
        order = deviation.argsort(dim=-1)                 # closest to typical first
        ordered = probs.gather(1, order)
        cumulative = ordered.cumsum(dim=-1)
        # Keep every token up to and including the one that crosses typical_p.
        drop_ordered = (cumulative - ordered) >= self.typical_p
        remove = torch.zeros_like(drop_ordered)
        remove.scatter_(1, order, drop_ordered)
        return remove


def truncation_warper(*, temperature=1.0, typical_p=None, min_p=None,
                      eta_cutoff=None):
    """The processor for whichever scheme was asked for, or None for neither."""
    if typical_p is None and min_p is None and eta_cutoff is None:
        return None
    return TruncationWarper(temperature=temperature, typical_p=typical_p,
                            min_p=min_p, eta_cutoff=eta_cutoff)
