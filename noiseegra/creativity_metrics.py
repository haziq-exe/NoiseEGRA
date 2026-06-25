from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SemanticDiversityResult:
    vendi_score: float
    semantic_score_mean: float
    semantic_score_std: float


@dataclass
class LexicalDiversityResult:
    self_bleu_mean: float
    self_bleu_std: float
    lexical_score_mean: float
    lexical_score_std: float


class CreativityScorer:
    """
    Computes semantic diversity, lexical diversity, and a combined creativity score.

    Semantic diversity:
    - Vendi Score over cosine-similarity kernel from normalized embeddings.
    - Normalized semantic score in [0, 1] using (vendi_score - 1) / (n - 1).

    Lexical diversity:
    - Self-BLEU (lower is more diverse), converted to lexical score as 1 - self_bleu, with stds.
    """

    _PAPER_MODAL_COLLAPSE_PATH = Path(__file__).resolve().parent / "data" / "paper_modal_collapse_indices.json"

    @classmethod
    def load_paper_modal_collapse_indices(cls) -> Dict[str, List[int]]:
        if not cls._PAPER_MODAL_COLLAPSE_PATH.is_file():
            return {}
        with cls._PAPER_MODAL_COLLAPSE_PATH.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        return {str(k): [int(i) for i in v] for k, v in raw.items()}

    def __init__(
        self,
        texts: Sequence[str],
        embedding_model: str = "BAAI/bge-m3",
        max_k: int = 10,
        random_state: int = 42,
    ):
        self.texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        self.model = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.max_k = max_k
        self.random_state = random_state
        self.total_modal_collapse_indices_by_run = self.load_paper_modal_collapse_indices()

    def _encode(self) -> np.ndarray:
        if not self.texts:
            raise ValueError("No valid texts were provided.")
        return self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

    @staticmethod
    def _safe_clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return float(max(low, min(high, value)))

    def change_text(self, texts: Sequence[str]) -> None:
        self.texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]

    def semantic_diversity(self) -> SemanticDiversityResult:
        embeddings = self._encode()
        n = embeddings.shape[0]

        # With normalized embeddings, dot-product equals cosine similarity.
        kernel = embeddings @ embeddings.T
        try:
            vendi_module = importlib.import_module("vendi_score.vendi")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'vendi_score'. Install it with: pip install vendi-score"
            ) from exc

        vendi_score = float(vendi_module.score_K(kernel))

        if n == 1:
            semantic_score_mean = 0.0
        else:
            semantic_score_mean = self._safe_clip((vendi_score - 1.0) / (n - 1.0))

        return SemanticDiversityResult(
            vendi_score=vendi_score,
            semantic_score_mean=semantic_score_mean,
            semantic_score_std=0.0,
        )

    @staticmethod
    def _extract_ngrams(tokens: Sequence[str], n: int) -> Dict[tuple, int]:
        counts: Dict[tuple, int] = {}
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i : i + n])
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    def _sentence_bleu(
        self,
        candidate: Sequence[str],
        references: Sequence[Sequence[str]],
        max_n: int = 4,
    ) -> float:
        if not candidate:
            return 0.0

        max_n = min(max_n, len(candidate))
        if max_n == 0:
            return 0.0

        precisions: List[float] = []
        for n in range(1, max_n + 1):
            cand_counts = self._extract_ngrams(candidate, n)
            total = sum(cand_counts.values())
            if total == 0:
                precisions.append(0.0)
                continue

            max_ref_counts: Dict[tuple, int] = {}
            for ref in references:
                ref_counts = self._extract_ngrams(ref, n)
                for ng, c in ref_counts.items():
                    max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), c)

            clipped = 0
            for ng, c in cand_counts.items():
                clipped += min(c, max_ref_counts.get(ng, 0))

            # add-1 smoothing
            precisions.append((clipped + 1.0) / (total + 1.0))

        ref_lens = [len(r) for r in references if r]
        cand_len = len(candidate)
        if not ref_lens:
            return 0.0

        closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))
        if cand_len > closest_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1.0 - (closest_ref_len / max(cand_len, 1)))

        log_precision = sum(math.log(max(p, 1e-12)) for p in precisions) / len(precisions)
        return bp * math.exp(log_precision)

    def lexical_diversity(self) -> LexicalDiversityResult:
        n = len(self.texts)
        if n < 2:
            return LexicalDiversityResult(
                self_bleu_mean=1.0,
                self_bleu_std=0.0,
                lexical_score_mean=0.0,
                lexical_score_std=0.0,
            )

        tokenized = [t.split() for t in self.texts]
        bleu_scores: List[float] = []

        for i, candidate in enumerate(tokenized):
            references = [tok for j, tok in enumerate(tokenized) if j != i]
            bleu_scores.append(self._sentence_bleu(candidate, references))

        self_bleu_mean = float(np.mean(bleu_scores))
        self_bleu_std = float(np.std(bleu_scores))

        lexical_score_mean = self._safe_clip(1.0 - self_bleu_mean)
        # lexical_score = 1 - self_bleu, so std is the same
        lexical_score_std = self_bleu_std

        return LexicalDiversityResult(
            self_bleu_mean=self_bleu_mean,
            self_bleu_std=self_bleu_std,
            lexical_score_mean=lexical_score_mean,
            lexical_score_std=lexical_score_std,
        )

    def creativity_score(
        self,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
        print_report: bool = True,
    ) -> float:
        total_weight = semantic_weight + lexical_weight
        if total_weight <= 0:
            raise ValueError("semantic_weight + lexical_weight must be > 0.")

        semantic = self.semantic_diversity()
        lexical = self.lexical_diversity()

        combined_mean = (
            semantic.semantic_score_mean * semantic_weight
            + lexical.lexical_score_mean * lexical_weight
        ) / total_weight

        # Approximate std via independent-error propagation (ignores covariance).
        combined_std = (
            (semantic.semantic_score_std * semantic_weight) ** 2
            + (lexical.lexical_score_std * lexical_weight) ** 2
        ) ** 0.5 / total_weight

        report = {
            "vendi_score": semantic.vendi_score,
            "semantic_diversity_score_mean": semantic.semantic_score_mean,
            "semantic_diversity_score_std": semantic.semantic_score_std,
            "self_bleu_mean": lexical.self_bleu_mean,
            "self_bleu_std": lexical.self_bleu_std,
            "lexical_diversity_score_mean": lexical.lexical_score_mean,
            "lexical_diversity_score_std": lexical.lexical_score_std,
            "combined_creativity_score_mean": float(combined_mean),
            "combined_creativity_score_std": float(combined_std),
        }

        if print_report:
            self.print_report(report)

        return float(combined_mean)

    def creativity_score_without_modal_collapse(
        self,
        run_type: str,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
        print_report: bool = True,
        exclude_indices: Optional[Sequence[int]] = None,
    ) -> float:
        """
        Compute creativity score after removing stories with TotalModalCollapse==1.

        Pass ``exclude_indices`` (0-based story indices) for new runs, or ``run_type``
        to use paper-replication indices from ``data/paper_modal_collapse_indices.json``.
        """
        if exclude_indices is not None:
            excluded_indices = set(int(i) for i in exclude_indices)
        elif run_type in self.total_modal_collapse_indices_by_run:
            excluded_indices = set(self.total_modal_collapse_indices_by_run[run_type])
        else:
            available = ", ".join(sorted(self.total_modal_collapse_indices_by_run.keys()))
            raise ValueError(
                f"Unknown run_type '{run_type}' and no exclude_indices provided. "
                f"Paper run types: {available}"
            )
        filtered_texts = [t for idx, t in enumerate(self.texts) if idx not in excluded_indices]

        if not filtered_texts:
            raise ValueError(
                f"All stories were removed for run_type '{run_type}'. "
                "Cannot compute creativity score."
            )

        original_texts = self.texts
        try:
            self.texts = filtered_texts
            return self.creativity_score(
                semantic_weight=semantic_weight,
                lexical_weight=lexical_weight,
                print_report=print_report,
            )
        finally:
            self.texts = original_texts

    def print_run_types(self) -> None:
        """Print available run_type keys discovered from SCORE files."""
        run_types = sorted(self.total_modal_collapse_indices_by_run.keys())
        if not run_types:
            print("No run_type keys found. Check scores_dir and SCORE files.")
            return

        print("=== Available run_type keys ===")
        for run_type in run_types:
            print(run_type)

    @staticmethod
    def print_report(report: Dict[str, float]) -> None:
        print("=== Creativity Report ===")
        print(
            "Semantic - Vendi Score: "
            f"{report['vendi_score']:.4f}"
        )
        print(
            "Semantic Diversity Score: "
            f"{report['semantic_diversity_score_mean']:.4f} "
            f"(std {report['semantic_diversity_score_std']:.4f})"
        )
        print(
            "Lexical - Self-BLEU: "
            f"{report['self_bleu_mean']:.4f} "
            f"(std {report['self_bleu_std']:.4f})"
        )
        print(
            "Lexical Diversity Score: "
            f"{report['lexical_diversity_score_mean']:.4f} "
            f"(std {report['lexical_diversity_score_std']:.4f})"
        )
        print(
            "Combined Creativity Score: "
            f"{report['combined_creativity_score_mean']:.4f} "
            f"(std {report['combined_creativity_score_std']:.4f})"
        )
