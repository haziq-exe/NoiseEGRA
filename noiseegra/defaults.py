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
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral-Nemo-12B": "mistralai/Mistral-Nemo-Instruct-2407",
}

# The paper injects into layers 12-20 of 32, i.e. the band from 37% to 62% of
# depth. These ranges put each model in the same relative band, 9 layers wide.
EN_MODEL_LAYER_RANGES: Dict[str, Tuple[int, int]] = {
    "Qwen3-8B": (14, 23),          # 36 blocks
    "Llama-3.1-8B": (12, 21),      # 32 blocks, identical to the paper
    "Mistral-Nemo-12B": (15, 24),  # 40 blocks
}

EN_CONSTRAINTS = ("closure", "present_tense", "simple_register", "dialogue")
EN_MAX_WORDS = 150
EN_MAX_GRADE_LEVEL = 6.0
