"""Paper-reported default hyperparameters and model settings."""

from __future__ import annotations

from typing import Dict, Tuple

# RMS noise calibration: std = RMS_ALPHA * median(block RMS)
RMS_ALPHA = 0.175

DEFAULT_MAX_NOISE_TOKENS = 200
DEFAULT_MAX_NEW_TOKENS_PLAN = 500
DEFAULT_MAX_NEW_TOKENS_STORY = 500

# Inclusive layer ranges used in the paper (Python range stop is exclusive)
MODEL_LAYER_RANGES: Dict[str, Tuple[int, int]] = {
    "ALLaM": (12, 21),
    "AceGPT": (12, 21),
    "Fanar": (18, 27),
    "Jais": (12, 21),
    "Phi-4-mini": (12, 21),
}

MODEL_HF_IDS: Dict[str, str] = {
    "ALLaM": "humain-ai/ALLaM-7B-Instruct-preview",
    "AceGPT": "FreedomIntelligence/AceGPT-v2-8B-Chat",
    "Fanar": "QCRI/Fanar-1-9B-Instruct",
    "Jais": "inceptionai/Jais-2-8B-Chat",
    "Phi-4-mini": "microsoft/Phi-4-mini-instruct",
}

BASELINE_NOISE_FREE = {"temperature": 1.0, "do_sample": True}
BASELINE_HIGH_TEMP_TOP_K = {"temperature": 1.8, "top_k": 40, "do_sample": True}
BASELINE_HIGH_TEMP_TOP_P = {"temperature": 1.8, "top_p": 0.9, "do_sample": True}

DEFAULT_EXPERIMENT_INPUT_FOLDERS = [
    "AENIMaxW",
    "baseline",
    "EmbedNoise",
    "ResidNoise",
    "AttnNoise",
]


# --------------------------------------------------------------------------- #
#  English generalisation: WritingPrompts + verifiable constraints             #
# --------------------------------------------------------------------------- #

# Standard dense, text-only causal LMs that run in float16 on 2x T4 (32 GB).
# Deliberately no hybrid/MoE/multimodal models: the method is defined on the
# residual stream of a standard transformer, and the original study used
# standard transformers, so changing architecture would confound the result.
EN_MODEL_HF_IDS: Dict[str, str] = {
    # Apache 2.0, ungated
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "OLMo-2-7B": "allenai/OLMo-2-1124-7B-Instruct",
    "Granite-3.1-8B": "ibm-granite/granite-3.1-8b-instruct",
    "Mistral-Nemo-12B": "mistralai/Mistral-Nemo-Instruct-2407",
    # Own licences, still ungated
    "Falcon3-7B": "tiiuae/Falcon3-7B-Instruct",
    # Gated: accept the licence on the model page first
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
}

# Block count per model, used to place the steering band.
EN_MODEL_DEPTHS: Dict[str, int] = {
    "Qwen3-8B": 36,
    "Qwen2.5-7B": 28,
    "OLMo-2-7B": 32,
    "Granite-3.1-8B": 40,
    "Mistral-Nemo-12B": 40,
    "Falcon3-7B": 28,
    "Llama-3.1-8B": 32,
}


def layer_band(n_layers: int, width: int = 9, end_frac: float = 0.625) -> Tuple[int, int]:
    """The paper's injection band, transferred to a model of any depth.

    The published runs use a 9-block band ending at roughly 62% of depth: blocks
    12-20 of 32 for the four 32-block models, and 18-26 of 42 for Fanar. This
    reproduces both exactly and extends the same rule to other depths. Returns a
    half-open (lo, hi) suitable for ``range``.
    """
    hi = int(round(end_frac * n_layers))
    lo = max(0, hi - width + 1)
    return lo, min(hi + 1, n_layers)


EN_MODEL_LAYER_RANGES: Dict[str, Tuple[int, int]] = {
    name: layer_band(depth) for name, depth in EN_MODEL_DEPTHS.items()
}

EN_CONSTRAINTS = ("closure", "present_tense", "simple_register", "dialogue")
EN_MAX_WORDS = 150
EN_MAX_GRADE_LEVEL = 6.0
