from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import math

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score


@dataclass
class SemanticDiversityResult:
    pairwise_dispersion_mean: float
    pairwise_dispersion_std: float
    cluster_count: int
    best_silhouette: float
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
    - Pairwise embedding dispersion (mean cosine distance) + std.
    - Cluster count selected by silhouette score over k-means.

    Lexical diversity:
    - Self-BLEU (lower is more diverse), converted to lexical score as 1 - self_bleu, with stds.
    """

    TOTAL_MODAL_COLLAPSE_INDICES_BY_RUN: Dict[str, List[int]] = {
        "ALLam__ATTN__L12-20__std0p105__maxtok200": [0, 1, 8, 10, 12, 13, 17, 30, 41, 46, 47, 48, 49],
        "ALLam__BASELINE": [],
        "ALLam__BASELINE__temp1p8__topk40": [],
        "ALLam__BASELINE__temp1p8__topp0p95": [0, 1, 2, 10, 12, 13],
        "ALLam__EMBED__std0p015": [11, 18, 19, 21, 32, 35, 36, 37, 38, 39, 42, 46, 48, 49],
        "ALLam__ENTROPY__L12-20__std0p105": [],
        "ALLam__L12-20__std0p036__decay0": [],
        "Fanar_9B__L18-26__std0p625__decay0": [],
        "Fanar__ATTN__L18-26__std4p095": [10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 30],
        "Fanar__BASELINE": [],
        "Fanar__BASELINE__temp1p8__topk40": [40],
        "Fanar__BASELINE__temp1p8__topp0p95": [],
        "Fanar__EMBED__std0p042": [10, 12, 15, 17, 19, 20, 21, 23, 25, 27, 29, 30],
        "Fanar__ENTROPY__L18-26__std4p095": [40],
        "JAIS__ATTN__L12-20__std4p6375": [10, 48],
        "Jais__BASELINE": [],
        "Jais__BASELINE__temp1p8__topk40": [],
        "Jais__BASELINE__temp1p8__topp0p95": [],
        "Jais__EMBED__std7p19426": [43],
        "Jais__ENTROPY__L12-20__std4p6375": [],
        "Jais__ENTROPY__L12-20__std9p275": [],
        "Jais__L12-20__std5p25__decay0": [],
        "PHI-4-MINI__ATTN__L12-20__std0p2975": [0, 3, 4, 7, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 38, 40],
        "PHI-4-MINI__BASELINE__temp1p8__topk40": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 46, 47, 49],
        "PHI-4-MINI__BASELINE__temp1p8__topp0p95": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        "PHI-4-MINI__L12-20__std0p177__decay0": [],
        "Phi-4__BASELINE": [],
        "Phi-4__ENTROPY__L12-20__std0p2975": [],
    }

    def __init__(
        self,
        texts: Sequence[str],
        embedding_model: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
        max_k: int = 10,
        random_state: int = 42,
    ):
        self.texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        self.model = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.max_k = max_k
        self.random_state = random_state
        # Copy to avoid accidental in-place edits affecting class-level constant.
        self.total_modal_collapse_indices_by_run = {
            k: list(v) for k, v in self.TOTAL_MODAL_COLLAPSE_INDICES_BY_RUN.items()
        }

    def _encode(self) -> np.ndarray:
        if not self.texts:
            raise ValueError("No valid texts were provided.")
        return self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    @staticmethod
    def _safe_clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return float(max(low, min(high, value)))

    def change_text(self, texts: Sequence[str]) -> None:
        self.texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]

    def semantic_diversity(self) -> SemanticDiversityResult:
        embeddings = self._encode()
        n = embeddings.shape[0]

        if n == 1:
            return SemanticDiversityResult(
                pairwise_dispersion_mean=0.0,
                pairwise_dispersion_std=0.0,
                cluster_count=1,
                best_silhouette=0.0,
                semantic_score_mean=0.0,
                semantic_score_std=0.0,
            )

        # Pairwise cosine distances (unique pairs only)
        dists = pairwise_distances(embeddings, metric="cosine")
        upper = dists[np.triu_indices(n, k=1)]
        pairwise_dispersion_mean = float(np.mean(upper))
        pairwise_dispersion_std = float(np.std(upper))

        best_k = 1
        best_sil = float("-inf")

        k_upper = min(self.max_k, n - 1)
        if n >= 3 and k_upper >= 2:
            for k in range(2, k_upper + 1):
                labels = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=self.random_state,
                ).fit_predict(embeddings)

                if len(set(labels)) < 2:
                    continue

                sil = float(silhouette_score(embeddings, labels, metric="cosine"))
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
        else:
            best_sil = 0.0

        if best_sil == float("-inf"):
            best_sil = 0.0

        # Normalized components for semantic score
        dispersion_norm = self._safe_clip(pairwise_dispersion_mean / 1.0)
        cluster_norm = self._safe_clip((best_k - 1) / max(n - 1, 1))

        semantic_score_mean = 0.5 * dispersion_norm + 0.5 * cluster_norm

        # Only dispersion component varies here; cluster_norm is deterministic given best_k.
        # dispersion_norm = clip(mean/2). We approximate std scaling by linear factor,
        # ignoring clipping effects (clipping can reduce variance near bounds).
        semantic_score_std = 0.5 * (pairwise_dispersion_std / 2.0)

        return SemanticDiversityResult(
            pairwise_dispersion_mean=pairwise_dispersion_mean,
            pairwise_dispersion_std=pairwise_dispersion_std,
            cluster_count=best_k,
            best_silhouette=best_sil,
            semantic_score_mean=semantic_score_mean,
            semantic_score_std=semantic_score_std,
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
            "pairwise_embedding_dispersion_mean": semantic.pairwise_dispersion_mean,
            "pairwise_embedding_dispersion_std": semantic.pairwise_dispersion_std,
            "cluster_count": float(semantic.cluster_count),
            "best_silhouette": semantic.best_silhouette,
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
    ) -> float:
        """
        Compute creativity score after removing stories that have TotalModalCollapse==1
        for the given run_type.
        """
        if run_type not in self.total_modal_collapse_indices_by_run:
            available = ", ".join(sorted(self.total_modal_collapse_indices_by_run.keys()))
            raise ValueError(
                f"Unknown run_type '{run_type}'. Available run types: {available}"
            )

        excluded_indices = set(self.total_modal_collapse_indices_by_run[run_type])
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
            "Semantic - Pairwise Embedding Dispersion: "
            f"{report['pairwise_embedding_dispersion_mean']:.4f} "
            f"(std {report['pairwise_embedding_dispersion_std']:.4f})"
        )
        print(f"Semantic - Cluster Count (k via silhouette): {int(report['cluster_count'])}")
        print(f"Semantic - Best Silhouette: {report['best_silhouette']:.4f}")
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
