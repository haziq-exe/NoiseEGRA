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
