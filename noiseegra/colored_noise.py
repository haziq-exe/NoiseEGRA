"""Noise with a 1/f^beta power spectrum, along the token axis.

This project has measured, repeatedly, that a perturbation which varies *within*
a story and is identically distributed *across* stories cannot make a set of
stories more varied: every story wobbles, they all wobble the same way, and they
end up no further apart. Per-token Gaussian noise -- the published method's own
mechanism -- is exactly that, and it is a null for variety here three times over.
A single fixed displacement per story is the opposite extreme: it moves stories
apart, and because it never lets go, it drags the whole story off register.

Those two are not a choice between mechanisms. They are the two ends of one
axis. Noise whose power spectral density falls as 1/f^beta is white at beta=0
(independent every token) and, as beta grows, puts more and more of its energy
in the lowest frequency, until at large beta the whole story sees essentially
one constant displacement. Everything between is a perturbation that drifts
slowly over the course of a story while still differing between stories.

The exponent is therefore a single continuous knob spanning "per token", "per
few tokens", "per sentence" and "per story", and it is the knob this project has
never turned. Reinforcement learning found the same axis worth turning: pink
noise (beta=1) beats white noise as an exploration signal for off-policy control
(Eberhard et al., ICLR 2023), and an exponent between white and pink is best
on-policy (Hollenstein et al., AAAI 2024). Their explanation transfers directly:
low-frequency energy makes a walk change direction less often and so explore
further from where it started, which is what a set of stories needs.

Generated in the frequency domain, following Timmer and Koenig (1995): draw a
complex Gaussian coefficient per frequency, scale it by f^(-beta/2), and
transform back.

What matters here, and what `tests/test_colored_noise.py` measures, is the split
of the total variance into the part that differs between stories -- the mean of
a story's trajectory -- and the part that only wobbles within one. That share is
what beta controls, and it is the only part that can move variety.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def colored_sequence(
    length: int,
    width: int = 1,
    *,
    beta: float = 1.0,
    fmin_cycles: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """`length` x `width` samples whose power spectrum falls as 1/f^beta.

    Each column is an independent sequence, scaled to unit variance over the
    whole block so that the size of the perturbation is the same whatever
    `beta` is -- otherwise an exponent sweep would be a magnitude sweep as well,
    which is how this project has produced a confounded result before.

    `fmin_cycles` is the slowest wobble the perturbation is allowed, counted in
    cycles across one story, and it decides what "constant for a whole story"
    can mean. At 1.0 the slowest component completes a full cycle within the
    story, so it averages to nothing over that story and no amount of `beta`
    produces a per-story displacement. Below 1.0 the slowest component covers
    only part of a cycle, which is a displacement that barely moves -- and as
    `beta` grows that term takes over and the mechanism becomes the fixed
    per-story displacement this project already has. beta=0 ignores the
    cut-off entirely and gives independent noise at every token, which is the
    published method's mechanism.
    """
    if length < 1:
        raise ValueError("length must be at least 1")
    if width < 1:
        raise ValueError("width must be at least 1")
    if beta < 0:
        raise ValueError("beta must not be negative; it is a rate of decay")

    n_freq = length // 2 + 1
    f = np.fft.rfftfreq(length)                      # 0 .. 0.5, length n_freq
    if fmin_cycles <= 0:
        raise ValueError("fmin_cycles must be positive")
    lowest = float(fmin_cycles) / length
    if lowest > 0.5:
        raise ValueError("fmin_cycles is larger than the sequence can carry")
    # The zero-frequency bin is what survives as beta grows, so it is given the
    # cut-off frequency's weight rather than being dropped. Dropping it -- the
    # usual convention, which forces zero mean -- would remove the only
    # component that differs between stories, and with it the entire effect
    # this mechanism exists to produce.
    f = np.maximum(f, lowest)
    scale = torch.from_numpy(f ** (-beta / 2.0)).to(dtype=torch.float64)

    shape = (n_freq, width)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    # A real-valued sequence needs a real coefficient at DC, and at Nyquist when
    # the length is even; otherwise the inverse transform throws the imaginary
    # part away and those bins come out with the wrong variance.
    imag[0, :] = 0.0
    if length % 2 == 0:
        imag[-1, :] = 0.0
    spectrum = torch.complex(real * scale.unsqueeze(1), imag * scale.unsqueeze(1))

    out = torch.fft.irfft(spectrum, n=length, dim=0)

    # Normalise by the ENSEMBLE standard deviation, computed in closed form,
    # never by each sequence's own spread over time. At large beta a sequence is
    # nearly constant, so its spread over time is nearly zero; dividing by that
    # would amplify the leftover wobble and throw away the constant -- removing
    # precisely the component that differs between stories. Getting this wrong
    # caps the between-story share at about a tenth however large beta is, which
    # reads exactly like the mechanism not working.
    #
    # irfft divides by `length`, and a coefficient at a frequency strictly
    # between DC and Nyquist is counted twice by the Hermitian symmetry, so it
    # contributes four times its squared scale.
    sq = (scale ** 2)
    total = sq[0].clone()
    if length % 2 == 0:
        total = total + sq[-1] + 4.0 * sq[1:-1].sum()
    else:
        total = total + 4.0 * sq[1:].sum()
    sd = float(torch.sqrt(total)) / length
    if sd > 0:
        out = out / sd
    return out.to(device=device, dtype=dtype)


def between_story_share(trajectories: torch.Tensor) -> float:
    """Fraction of the total variance carried by whole-story means.

    `trajectories` is stories x length (x width). A value near 0 means the
    perturbation only wobbles within a story and every story sees the same
    distribution of wobbles, which cannot move variety. A value near 1 means
    each story sees one displacement of its own, which can.
    """
    x = trajectories.double()
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError("expected stories x length (x width)")
    if x.shape[0] < 2:
        raise ValueError("need at least two stories to split the variance")
    # Average over TIME only. Averaging over the independent coordinates as well
    # mixes together perturbations that have nothing to do with each other and
    # divides the answer by roughly the number of coordinates, which reads as
    # the mechanism saturating when it is doing exactly what it should.
    per_story_mean = x.mean(dim=1)                       # stories x width
    between = per_story_mean.var(dim=0, unbiased=False)  # width
    total = x.reshape(-1, x.shape[-1]).var(dim=0, unbiased=False)
    ok = total > 0
    if not bool(ok.any()):
        return 0.0
    return float((between[ok] / total[ok]).mean())
