from __future__ import annotations

import csv
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class ExperimentSpec:
    use_residual_noise: bool = False
    residual_layers: Optional[Sequence[int]] = None
    residual_noise_std: float = 0.0
    residual_noise_decay: float = 0.0
    max_noise_tokens: int = 250

    logits_noise_std: float = 0.0
    logits_noise_decay: float = 0.0
    max_new_tokens_plan: int = 500
    max_new_tokens_story: int = 500
    temperature: float = 1.0


def _seed_for_story(x: int) -> int:
    return 42 * (x**7) * 217


def _append_csv(path: Path, row: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(list(row))


def _make_plan_prompt():
    return [
        {"role": "system", "content": prompts.SYS_NOISE},
        {"role": "user", "content": prompts.NOISE_1},
    ]


def _make_story_prompt(plan_text: str):
    return [
        {"role": "system", "content": prompts.SYS_NOISE},
        {"role": "user", "content": prompts.NOISE_3.replace("[STORY PLAN]", plan_text)},
    ]

def _story_prompt():
    return [
        {"role": "system", "content": prompts.SYS_ZERO_SHOT},
        {"role": "user", "content": prompts.PROMPT_ZERO_SHOT},
    ]


def _layers_tag(layers: Optional[Sequence[int]]) -> str:
    if not layers:
        return "Lnone"
    layers = list(layers)
    return f"L{min(layers)}-{max(layers)}"


def _float_tag(x: float) -> str:
    # stable filesystem-friendly formatting
    s = f"{x:.6g}"          # up to 6 sig figs
    return s.replace(".", "p").replace("-", "m")


def _spec_to_run_id(model_name: str, spec: ExperimentSpec) -> str:
    if not spec.use_residual_noise:
        return f"{model_name}__BASELINE"
    return (
        f"{model_name}__{_layers_tag(spec.residual_layers)}"
        f"__std{_float_tag(spec.residual_noise_std)}"
        f"__decay{_float_tag(spec.residual_noise_decay)}"
    )


def run_story_experiments(
    model: Any,
    model_name: str,
    num_stories,
    specs: Sequence[ExperimentSpec],
    *,
    output_dir: str = "results",
    clear_cuda_each_iter: bool = True,
) -> dict[str, list[str]]:
    """
    Auto filenames: {run_id}.csv where run_id encodes model + spec.
    Returns: {run_id: [story_text, ...]}.
    """
    checker = EGRAConstraintChecker()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build run_ids and output paths (no hashing specs)
    run_ids: list[str] = [_spec_to_run_id(model_name, s) for s in specs]
    out_paths: dict[str, Path] = {rid: out_dir / f"{rid}.csv" for rid in run_ids}

    outputs: dict[str, list[str]] = {rid: [] for rid in run_ids}

    # ---- EXPERIMENT SUMMARY (printed once) ----
    print("\n================ RUN CONFIGURATION ================\n")
    for spec, rid in zip(specs, run_ids):
        print(f"RUN ID: {rid}")
        if not spec.use_residual_noise:
            print("  type: BASELINE (no residual noise)")
        else:
            print("  type: RESIDUAL NOISE")
            print(f"  residual_layers: {list(spec.residual_layers or [])}")
            print(f"  residual_noise_std: {spec.residual_noise_std}")
            print(f"  residual_noise_decay: {spec.residual_noise_decay}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        print()

    print("===================================================\n")

    start_story = num_stories[0]
    end_story = num_stories[1]
    for x in range(start_story, end_story):
        seed = _seed_for_story(x)

        for spec, rid in zip(specs, run_ids):
            # 1) plan
            # plan_prompt = _make_plan_prompt()
            story_prompt = _story_prompt()
            if spec.use_residual_noise:
                story_text = model.generate_with_residual_stream_noise(
                    story_prompt,
                    residual_layers=list(spec.residual_layers or []),
                    residual_noise_std=spec.residual_noise_std,
                    residual_noise_decay=spec.residual_noise_decay,
                    max_noise_tokens = spec.max_noise_tokens,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_plan,
                    temperature=spec.temperature,
                    seed=seed,
                )
            #     plan_text = model.generate_with_residual_stream_noise(
            #         plan_prompt,
            #         residual_layers=list(spec.residual_layers or []),
            #         residual_noise_std=spec.residual_noise_std,
            #         residual_noise_decay=spec.residual_noise_decay,
            #         logits_noise_std=spec.logits_noise_std,
            #         logits_noise_decay=spec.logits_noise_decay,
            #         max_new_tokens=spec.max_new_tokens_plan,
            #         temperature=spec.temperature,
            #         seed=seed,
            #     )
            else:
                story_text = model.generate(
                    story_prompt,
                    max_new_tokens=spec.max_new_tokens_plan,
                    seed=seed,
                )

            # 2) story
            # story_prompt = _make_story_prompt(plan_text)
            # story_text = model.generate(
            #     story_prompt,
            #     max_new_tokens=spec.max_new_tokens_story,
            #     seed=seed,
            # )

            outputs[rid].append(story_text)
            _append_csv(out_paths[rid], [story_text])

        if clear_cuda_each_iter:
            gc.collect()
            torch.cuda.empty_cache()

    # scoring + constraints
    for rid in run_ids:
        stories = outputs[rid]
        print(f"\n\n---- {rid} ----\n")
        CreativityScorer(stories).creativity_score(print_report=True)

        print(f"\n\n------------------ {rid} CONSTRAINT ---------------------\n\n\n")
        checker.print_report(stories)

    return outputs


def make_specs(*items: Any) -> list[ExperimentSpec]:
    """
    Accepts:
      - dicts with residual parameters
      - "zero_shot"/"baseline"
    """
    specs: list[ExperimentSpec] = []

    for it in items:
        if isinstance(it, str) and it.lower() in {"zero_shot", "baseline"}:
            specs.append(ExperimentSpec(use_residual_noise=False))
            continue

        if isinstance(it, Mapping):
            specs.append(
                ExperimentSpec(
                    use_residual_noise=True,
                    residual_layers=it.get("residual_layers"),
                    residual_noise_std=float(it.get("residual_noise_std", 0.0)),
                    residual_noise_decay=float(it.get("residual_noise_decay", 0.0)),
                    max_noise_tokens=float(it.get("max_noise_tokens", 250.0)),
                    logits_noise_std=float(it.get("logits_noise_std", 0.0)),
                    logits_noise_decay=float(it.get("logits_noise_decay", 0.0)),
                    max_new_tokens_plan=int(it.get("max_new_tokens_plan", 1000)),
                    max_new_tokens_story=int(it.get("max_new_tokens_story", 500)),
                    temperature=float(it.get("temperature", 1.0)),
                )
            )
            continue

        raise TypeError(f"Unsupported spec type: {type(it)}")
    return specs