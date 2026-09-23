"""Published diversity methods, run on the same task as the headline method.

Each gets exactly the instruction the headline method gets -- the same story
request with the same eight rules -- and writes stories for the same runner, so
they are checkpointed, split across GPUs, checked for coherence, scored on the
rules and judged for sameness the same way. Methods that make stories in
groups (several in one reply, a conversation, a batch written together) make a
whole group when its first story is asked for and hand the rest out in order.

- ``verbalized``: Verbalized Sampling (Zhang et al., ICML 2026), VS-Standard, the
  paper's main variant, with its own prompt text from the authors' repository
  (github.com/CHATS-lab/verbalized-sampling, ``methods/prompt.py``): k responses
  with verbalized probabilities in one JSON reply, k = 5, every response used.
- ``ssot``: String Seed of Thought (Misaki & Akiba, ICLR 2026), the
  diversity-aware system prompt of the paper's Listing A.3: write a random
  string, reason from it, answer inside ``<answer>`` tags.
- ``incontext``: in-context regeneration, the strongest prompting method in
  NoveltyBench (Zhang et al., COLM 2025) and the "Diverse Prompt" baseline of G2
  (EMNLP 2025): ten stories in one conversation, each after "Can you generate a
  different answer?" with the earlier ones in context, as the benchmark's code
  does it (``src/inference.py``).
- ``stars``: STARS (ICLR 2026), the authors' code (github.com/lythk88/STARS,
  ``liveideabench/steering.py``) ported unchanged: the stories of a batch are
  written together and, at every token, a one-step Stiefel update pushes their
  inputs to one layer's attention output projection apart. Layer 20, C = 0.1,
  twenty stories a batch (the paper's settings for Qwen3-1.7B).
- ``noiseinject``: noise-enhanced sampling (Liu et al., ICLR 2026, Algorithm
  1): one draw of uniform noise U(0, alpha)^d per story, added to the MLP output
  of the top third of layers (20-27 of 28, the paper's range for a 28-layer
  model), the same draw at every layer and position. alpha = 0.07, the paper's
  main setting.

Every method samples at the run's temperature with no top-p or top-k cut, the
same decoding the headline method uses; none of them is steered toward the rules.
"""

from __future__ import annotations

import json
import re
import zlib
from typing import Dict, List, Optional

import torch

from .EGRA_functions import strip_reasoning

# Ordered so that, split alternately over two GPUs with min-p and the steering
# combination first, the two cards carry about the same generation time.
METHODS = ("verbalized", "ssot", "stars", "incontext", "noiseinject")

NAMES = {
    "verbalized": "Verbalized Sampling (ICML 2026)",
    "ssot": "String Seed of Thought (ICLR 2026)",
    "incontext": "In-context regeneration (NoveltyBench, COLM 2025)",
    "stars": "STARS activation steering (ICLR 2026)",
    "noiseinject": "Noise injection, Liu et al. (ICLR 2026)",
}

# The group each method writes at once. Story k belongs to group k // size.
GROUP = {"verbalized": 5, "incontext": 10, "stars": 20}

_CACHE: Dict[tuple, List[str]] = {}


def _seed(method: str, group: int) -> int:
    return zlib.crc32(f"{method}:{group}".encode()) & 0x7FFFFFFF


def _messages_key(messages) -> int:
    return zlib.crc32(json.dumps(messages, sort_keys=True).encode())


# --------------------------------------------------------------------------- #
#  Verbalized Sampling                                                         #
# --------------------------------------------------------------------------- #

def vs_system(k: int, words: int) -> str:
    """VS-Standard's system text for creative writing, verbatim from the repository:
    the standard prompt, then the vs_standard format with the default ('implicit')
    probability definition and no probability tuning."""
    return (f"\nGenerate {k} responses to the input prompt. Each response should be "
            f"approximately {words} words.\n"
            "\nReturn the responses in JSON format with the key: \"responses\" (list of "
            "dicts). Each dictionary must include:\n"
            "- 'text': the response string only (no explanation or extra text).\n"
            "- 'probability': how likely this response would be (from 0.0 to 1.0).\n"
            "\nRandomly sample the responses from the full distribution. Return ONLY the "
            "JSON object, with no additional explanations or text.\n")


def parse_vs(text: str) -> List[str]:
    """The 'text' fields of a VS reply, tolerating the JSON being cut short."""
    out: List[str] = []
    lo, hi = text.find("{"), text.rfind("}")
    if 0 <= lo < hi:
        try:
            obj = json.loads(text[lo:hi + 1])
            items = obj.get("responses", []) if isinstance(obj, dict) else obj
            for it in items if isinstance(items, list) else []:
                if isinstance(it, dict) and isinstance(it.get("text"), str):
                    out.append(it["text"])
                elif isinstance(it, str):
                    out.append(it)
            if out:
                return out
        except (ValueError, AttributeError):
            pass
    for m in re.finditer(r'["\']text["\']\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S):
        try:
            out.append(json.loads('"' + m.group(1) + '"'))
        except ValueError:
            out.append(m.group(1))
    return out


def _verbalized(egra, messages, params, group, temperature, max_new_tokens) -> List[str]:
    k = int(params.get("k", 5))
    words = int(params.get("words", 150))
    sys_text = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = [m for m in messages if m["role"] != "system"]
    msgs = [{"role": "system", "content": sys_text + "\n" + vs_system(k, words)}] + user
    got: List[str] = []
    tries = int(params.get("tries", 3))
    for attempt in range(tries):
        raw = egra.generate(msgs, max_new_tokens=int(params.get("max_new_tokens", 2400)),
                            do_sample=True, temperature=temperature,
                            seed=_seed("verbalized", group) + attempt)
        got = [t.strip() for t in parse_vs(raw) if t and t.strip()]
        if got:
            break
    print(f"  [verbalized] group {group}: {len(got)} of {k} responses parsed"
          + ("" if got else " -- none after retries; counted as failed stories"), flush=True)
    return (got + [""] * k)[:k]


# --------------------------------------------------------------------------- #
#  String Seed of Thought                                                      #
# --------------------------------------------------------------------------- #

SSOT_SYSTEM = (
    "You are a helpful AI Assistant designed to provide well-reasoned and detailed "
    "responses. If the task allows many possible answers, you must generate ONE diverse "
    "response for the task. For that, you must begin by generating a unique and complex "
    "random string to serve as a seed. This random string should appear sufficiently "
    "complex and unpredictable, with no obvious structure or pattern. Use your judgment to "
    "ensure it looks arbitrary and unguessable.\n\n"
    "If the user asks you some question which allows multiple answers, use the generated "
    "seed (the exact contents inside the <random_string> tags) to guide any random "
    "sampling or stochastic decisions.\n\n"
    "Follow these steps for every instruction:\n"
    "1. Output the random seed string enclosed within <random_string> and </random_string> "
    "tags.\n"
    "2. Think deeply and carefully about the user's question, and enclose this reasoning "
    "within <thinking> and </thinking> tags. You have to generate ONE response leveraging "
    "the generated seed—the exact contents inside the <random_string> tags, to ensure "
    "your single answer is unique and diverse. Make sure to extract maximum randomness "
    "from the string by using all of its content.\n"
    "3. Provide your final answer, enclosed within <answer> and </answer> tags.\n\n"
    "Strictly follow this tag structure, and respond in the following format:\n"
    "<random_string>\n...\n</random_string>\n<thinking>\n...\n</thinking>\n"
    "<answer>\n...\n</answer>"
)


def parse_ssot(text: str) -> str:
    """The final answer: inside the last <answer> tags, else after </thinking>."""
    m = list(re.finditer(r"<answer>(.*?)(?:</answer>|$)", text, re.S | re.I))
    if m:
        return m[-1].group(1).strip()
    i = text.lower().rfind("</thinking>")
    if i >= 0:
        return text[i + len("</thinking>"):].strip()
    return text.strip()


def _ssot(egra, messages, params, seed, temperature, max_new_tokens) -> str:
    sys_text = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = [m for m in messages if m["role"] != "system"]
    msgs = [{"role": "system", "content": SSOT_SYSTEM + "\n\n" + sys_text}] + user
    raw = egra.generate(msgs, max_new_tokens=int(params.get("max_new_tokens", 1500)),
                        do_sample=True, temperature=temperature, seed=seed)
    return parse_ssot(raw)


# --------------------------------------------------------------------------- #
#  In-context regeneration                                                     #
# --------------------------------------------------------------------------- #

AGAIN = "Can you generate a different answer?"


def _incontext(egra, messages, params, group, temperature, max_new_tokens, max_words
               ) -> List[str]:
    n = int(params.get("n", 10))
    msgs = list(messages)
    out: List[str] = []
    for i in range(n):
        text = egra.generate(msgs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, seed=_seed("incontext", group) + i,
                             max_words=max_words)
        out.append(text)
        msgs = msgs + [{"role": "assistant", "content": text},
                       {"role": "user", "content": AGAIN}]
    return out


# --------------------------------------------------------------------------- #
#  STARS                                                                       #
# --------------------------------------------------------------------------- #

def svd_one_step(H, C=0.1, eps=0):
    """The authors' one-step Stiefel update (liveideabench/steering.py), unchanged
    apart from building the square root as a tensor op."""
    H = H.T  # now H is d x N
    d, N = H.shape
    Q, Sigma, RT = torch.linalg.svd(H, full_matrices=True)
    alpha = C * (Sigma[0] ** 2)
    alpha_sqrt = torch.sqrt(alpha)
    r = torch.sum(Sigma > eps).item()
    Sigma_r = Sigma[:r]
    Q2 = Q[:, r:d]
    indices = torch.randperm(d - r)[:N]
    indices, _ = torch.sort(indices)
    V0 = alpha_sqrt * Q2[:, indices.to(Q2.device)]
    _D = Sigma_r ** 2 / (Sigma_r ** 2 + alpha)
    D1 = torch.sum(_D) * 2
    D2 = torch.sum(_D ** 2) * 4
    eta_opt = D1 / D2
    t_Sigma = 1 / torch.sqrt(alpha + eta_opt ** 2 * Sigma)
    V1 = alpha_sqrt * (V0 + eta_opt * H) @ RT.T @ torch.diag(t_Sigma) @ RT
    return V1


def _stars_hook(module_to_hook, C: float, every: int = 1):
    """The authors' forward pre-hook on the attention output projection."""
    state = {"count": 0, "v": None}

    def hook(module, inp):
        state["count"] += 1
        x = inp[0] if isinstance(inp, tuple) else inp
        rest = inp[1:] if isinstance(inp, tuple) else ()
        c = state["count"]
        if (c - 2) % every == 0 or c == 2:
            h_last = x[:, -1, :].detach().to(torch.float32)
            state["v"] = svd_one_step(h_last, C=C).to(dtype=x.dtype, device=x.device)
        if state["v"] is not None:
            x[:, -1, :] += state["v"].T
        return (x, *rest) if isinstance(inp, tuple) else x

    return module_to_hook.register_forward_pre_hook(hook)


def _stars(egra, messages, params, group, temperature, max_new_tokens) -> List[str]:
    n = int(params.get("n", 20))
    blocks = egra._get_transformer_blocks()
    layer = min(int(params.get("layer", 20)), len(blocks) - 1)
    torch.manual_seed(_seed("stars", group))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed("stars", group))
    chat = egra.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = egra.tokenizer(chat, return_tensors="pt").to(egra._input_device())
    inputs.pop("token_type_ids", None)
    kw = egra._sampling_kwargs(do_sample=True, temperature=temperature)
    handle = _stars_hook(blocks[layer].self_attn.o_proj, float(params.get("C", 0.1)))
    try:
        out = egra.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  num_return_sequences=n, **kw)
    finally:
        handle.remove()
    start = inputs["input_ids"].shape[-1]
    return [strip_reasoning(egra.tokenizer.decode(row[start:], skip_special_tokens=True))
            for row in out]


# --------------------------------------------------------------------------- #
#  Noise injection                                                             #
# --------------------------------------------------------------------------- #

def _noiseinject(egra, messages, params, seed, temperature, max_new_tokens, max_words
                 ) -> str:
    blocks = egra._get_transformer_blocks()
    lo, hi = int(params.get("lo", 20)), int(params.get("hi", 28))
    lo, hi = min(lo, len(blocks) - 1), min(hi, len(blocks))
    alpha = float(params.get("alpha", 0.07))
    dim = egra.model.config.hidden_size
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    eps = torch.rand(dim, generator=g) * alpha           # one draw of U(0, alpha)^d

    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            return out + eps.to(device=out.device, dtype=out.dtype)
        return out

    handles = [blocks[l].mlp.register_forward_hook(hook) for l in range(lo, hi)]
    try:
        return egra.generate(messages, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, seed=seed, max_words=max_words)
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def generate_prior(egra, method: str, params: dict, messages, *, seed: int, story_index: int,
                   max_new_tokens: int, max_words: Optional[int] = None,
                   temperature: float = 1.0) -> str:
    """Story ``story_index`` of ``method`` on ``messages``."""
    if method not in METHODS:
        raise ValueError(f"unknown prior method {method!r}; one of {METHODS}")
    if method == "ssot":
        return _ssot(egra, messages, params, seed, temperature, max_new_tokens)
    if method == "noiseinject":
        return _noiseinject(egra, messages, params, seed, temperature, max_new_tokens,
                            max_words)
    size = int(params.get("k" if method == "verbalized" else "n", GROUP[method]))
    group, slot = divmod(int(story_index), size)
    key = (method, tuple(sorted(params.items())), _messages_key(messages), temperature, group)
    if key not in _CACHE:
        if method == "verbalized":
            _CACHE[key] = _verbalized(egra, messages, params, group, temperature,
                                      max_new_tokens)
        elif method == "incontext":
            _CACHE[key] = _incontext(egra, messages, params, group, temperature,
                                     max_new_tokens, max_words)
        else:
            _CACHE[key] = _stars(egra, messages, params, group, temperature, max_new_tokens)
    return _CACHE[key][slot]
