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
    # Small models, where the constraints actually bite. Qwen3-8B satisfies three
    # of the five steered requirements at 88-100% unsteered, which leaves a
    # steering direction nothing to win: the measured effect of pushing a direction
    # that controls an already-satisfied property is collateral damage and nothing
    # else. A model that fails those requirements gives the direction room to show
    # whether it helps.
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
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
    "Qwen3-1.7B": 28,
    "Qwen2.5-1.5B": 28,
    "Llama-3.2-1B": 16,
    "SmolLM2-1.7B": 24,
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

# The directions steering is applied along. Only four, because a steering vector
# has to be extractable from minimal pairs: "wrap the story up" is a direction,
# "use no digits" is not.
# Steering is applied along the constraints whose violation is a *local* property
# of the text -- tense, register, quoted speech, sentence length, how a sentence
# starts. A counting constraint ("between 50 and 65 words", "exactly one name") has
# no token-level direction that means "stop at 65", so it is asked for in the
# prompt and scored, but not steered. The earlier list steered `closure`, which was
# not one of the scored requirements at all.
EN_STEER_VECTORS = ("present_tense", "simple_register", "dialogue", "terse",
                    "varied_openers")

# What the prompt asks for and the scorer checks. Twelve, matching the pressure
# the original EGRA prompt put on the model, and every one of them decidable
# without a judge. See noiseegra.constraint_metrics_en.
EN_TASK_CONSTRAINTS = (
    "length", "present_tense", "simple_register", "dialogue",
    "easy_opening", "sentence_band", "sentence_count", "short_words",
    "one_name", "varied_openers", "plain_punctuation", "spelled_number",
    "no_repetition",
)

# Back-compatible alias: older callers used this for the steering names.
EN_CONSTRAINTS = EN_STEER_VECTORS

EN_MIN_WORDS = 50
EN_MAX_WORDS = 65
EN_MAX_GRADE_LEVEL = 2.5
