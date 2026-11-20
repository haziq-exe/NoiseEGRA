"""Checks that write no new stories, behind the 2026-11-20 maths survey.

1. Spread: do the per-story key changes send the planning state in many
   directions or a few? For 50 story seeds, the planning state's movement under
   each key change; participation ratio of the 50 movements (how many
   independent directions they fill, at most 50), share of their squared length
   on the top 1/3/5 directions, mean |cosine| between pairs. References: random
   key noise, and the gradient toward a uniformly random plan direction (whose
   movements also give a rough estimate of how fast the prompt-to-plan
   sensitivity falls off). For the sampled-alternative change, which first
   words the 50 seeds aim at (repeats limit how many stories can differ).
2. Cost: for stories already written with a key change removed after N tokens,
   how far the changed model is from the untouched one on that story's own
   words -- the log-likelihood ratio summed over the opening (nats), the full
   next-word KL at every position of the first 64 story tokens, and where a
   "release when the two agree" rule would have fired.
3. Suppressed or gone: the base model (before instruction tuning) against the
   chat model on the first word and on short openings -- how the chat model's
   log-probabilities relate to the base model's, and how much base-likely mass
   the chat model has made all but impossible.
"""
import gc
import json
import math
import re
import zlib

import torch

from .attn_space import (N_PLAN, _apply_key_delta, _plan_layer, _plan_state, attach,
                         key_gradient_push)

BASE_IDS = {"OLMo-3-7B": "allenai/Olmo-3-1025-7B", "Qwen3-8B": "Qwen/Qwen3-8B-Base"}
SETTINGS = {"OLMo-3-7B": [("real", 0.04), ("alt", 0.03)],
            "Qwen3-8B": [("alt", 0.02), ("real", 0.04)]}
PREFIX = {"OLMo-3-7B": "OLMo", "Qwen3-8B": "Qwen"}
KIND = {"real-state": "real", "sampled-alternative": "alt"}


def _spread(M):
    """M: n x D movements. Participation ratio, top-1/3/5 share, mean |cos|, eigenvalues."""
    M = M.double()
    ev = torch.linalg.eigvalsh(M @ M.T).clamp_min(0).flip(0)
    pr = float(ev.sum() ** 2 / ev.pow(2).sum().clamp_min(1e-30))
    share = [float(ev[:k].sum() / ev.sum()) for k in (1, 3, 5)]
    U = M / M.norm(dim=1, keepdim=True).clamp_min(1e-30)
    C = (U @ U.T).abs()
    iu = torch.triu_indices(len(M), len(M), 1)
    return {"participation_ratio": pr, "top_share_1_3_5": share,
            "mean_abs_cos": float(C[iu[0], iu[1]].mean()), "eig": [float(e) for e in ev[:20]]}


def _first_word_logprobs(model, ids):
    with torch.no_grad():
        o = model.model(input_ids=ids, use_cache=False)
    return torch.log_softmax(o.logits[0, -1].float(), dim=-1).cpu()


def _alt_target(lp, seed):
    """The first word the sampled-alternative change aims at for this seed (as in key_gradient_push)."""
    pr = lp.exp().clone()
    pr[int(pr.argmax())] = 0
    g = torch.Generator(device="cpu").manual_seed((int(seed) * 1000003 + zlib.crc32(b"alt")) & 0x7FFFFFFF)
    return int(torch.multinomial(pr / pr.sum(), 1, generator=g))


def spread_check(model, ids, layers, seeds, settings):
    blocks = model._get_transformer_blocks()
    tgt = min(_plan_layer(blocks), len(blocks) - 1)
    with torch.no_grad():
        clean = _plan_state(model, ids, tgt).float()
    scale = float(clean.norm(dim=-1).mean())
    n_prompt = int(ids.shape[-1])

    def moved(handles):
        try:
            with torch.no_grad():
                return (_plan_state(model, ids, tgt).float() - clean).flatten().cpu()
        finally:
            for h in handles:
                h.remove()
    res = {}
    arms = [(f"{obj} {size}", lambda s, o=obj, z=size: _apply_key_delta(
        model, key_gradient_push(model, ids, layers, s, o, z))) for obj, size in settings]
    arms += [("random-plan-gradient 0.02", lambda s: _apply_key_delta(
        model, key_gradient_push(model, ids, layers, s, "plan", 0.02))),
             ("key-noise 0.5", lambda s: attach(model, layers, n_prompt, s, spec={"k": 0.5})[0])]
    for name, make in arms:
        M = torch.stack([moved(make(s)) for s in seeds])
        r = _spread(M)
        r["moved_rel"] = float(M.view(len(seeds), N_PLAN, -1).norm(dim=-1).mean()) / scale
        if name.startswith("random-plan-gradient"):
            # movement ~ J J^T r for isotropic r: eigenvalues ~ sigma^4
            ev = torch.tensor(r["eig"]).clamp_min(0)
            r["sigma_ratio_estimate"] = [float((e / ev[0]) ** 0.25) for e in ev[:10]]
        res[name] = r
        print(f"  [spread] {name:28s} participation {r['participation_ratio']:5.2f} of {len(seeds)}, "
              f"top 1/3/5 share {r['top_share_1_3_5'][0]:.2f}/{r['top_share_1_3_5'][1]:.2f}/"
              f"{r['top_share_1_3_5'][2]:.2f}, mean |cos| {r['mean_abs_cos']:.3f}, "
              f"plan moved {r['moved_rel']:.3f} x its size", flush=True)
    lp = _first_word_logprobs(model, ids)
    tok = model.tokenizer
    ys = [_alt_target(lp, s) for s in seeds]
    blocks10 = [len(set(ys[i:i + 10])) for i in range(0, len(ys) - 9, 10)]
    counts = {}
    for y in ys:
        counts[y] = counts.get(y, 0) + 1
    top = int(lp.argmax())
    res["alt_targets"] = {
        "favourite": [tok.decode([top]), float(lp[top].exp())],
        "distinct_per_10": blocks10, "distinct_of_all": len(counts),
        "targets": sorted([[tok.decode([y]), c, float(lp[y].exp())] for y, c in counts.items()],
                          key=lambda t: -t[1])}
    print(f"  [spread] favourite first word {tok.decode([top])!r} p={float(lp[top].exp()):.3f}; "
          f"sampled-alternative targets distinct per 10 seeds {blocks10}, of {len(ys)}: {len(counts)}; "
          f"most common {res['alt_targets']['targets'][:6]}", flush=True)
    return res, lp


def _parse_arm(name):
    m = re.search(r"(real-state|sampled-alternative) ([\d.]+) removed after (\d+)", name)
    return (KIND[m.group(1)], float(m.group(2)), int(m.group(3))) if m else None


def cost_check(model, ids, layers, seeds, arms, n_stories=25, T=64):
    tok = model.tokenizer
    n_prompt = int(ids.shape[-1])
    res = {}
    for name, stories in arms.items():
        p = _parse_arm(name)
        if p is None:
            continue
        obj, size, n_on = p
        costs, kls, curves, release = [], [], [], []
        for x, story in enumerate(stories[:n_stories]):
            s = tok(story, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :T].to(ids.device)
            if s.shape[-1] < 8:                      # empty or broken story: nothing to score
                continue
            full = torch.cat([ids, s], 1)
            with torch.no_grad():
                clean = model.model(input_ids=full, use_cache=False).logits[0, n_prompt - 1:-1].float()
            hs = _apply_key_delta(model, key_gradient_push(model, ids, layers, seeds[x], obj, size))
            try:
                with torch.no_grad():
                    ch = model.model(input_ids=full, use_cache=False).logits[0, n_prompt - 1:-1].float()
            finally:
                for h in hs:
                    h.remove()
            lc, lq = clean.log_softmax(-1), ch.log_softmax(-1)
            y = s[0].view(-1, 1)
            lr = (lq.gather(-1, y) - lc.gather(-1, y)).squeeze(-1).cpu()
            kl = (lq.exp() * (lq - lc)).sum(-1).clamp_min(0).cpu()
            n = min(n_on, len(lr))
            costs.append(float(lr[:n].sum()))
            kls.append(float(kl[:n].sum()))
            curves.append(kl)
            r = None
            for t in range(4, len(kl)):
                if float(kl[t - 3:t + 1].mean()) < 0.25 * float(kl[0]):
                    r = t + 1
                    break
            release.append(r)
            del clean, ch, lc, lq
        if not curves:
            continue
        L = min(len(c) for c in curves)
        curve = torch.stack([c[:L] for c in curves]).mean(0)
        fired = sorted(r for r in release if r is not None)
        res[name] = {"opening_logratio_nats": sum(costs) / len(costs),
                     "opening_kl_nats": sum(kls) / len(kls),
                     "per_story_logratio": costs,
                     "kl_curve": [float(v) for v in curve],
                     "release_rule": {"fired": len(fired), "of": len(release),
                                      "median": fired[len(fired) // 2] if fired else None,
                                      "all": release}}
        pts = {t: float(curve[t]) for t in (0, 4, 8, 16, 32, L - 1) if t < L}
        print(f"  [cost] {name:48s} opening cost {res[name]['opening_logratio_nats']:6.2f} nats "
              f"(expected {res[name]['opening_kl_nats']:6.2f}); KL by position "
              + " ".join(f"{t}:{v:.3f}" for t, v in pts.items())
              + f"; release rule fired {len(fired)}/{len(release)}, median "
              f"{res[name]['release_rule']['median']}", flush=True)
    return res


def _score(model, tok, prefix_ids, texts, T=16):
    """Sum and count of log p(text's first T tokens | prefix)."""
    out = []
    dev = prefix_ids.device
    for t in texts:
        s = tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :T].to(dev)
        full = torch.cat([prefix_ids, s], 1)
        with torch.no_grad():
            lg = model(input_ids=full, use_cache=False).logits[0, prefix_ids.shape[-1] - 1:-1].float()
        lp = lg.log_softmax(-1).gather(-1, s[0].view(-1, 1)).squeeze(-1)
        out.append([float(lp.sum()), int(s.shape[-1])])
    return out


def plain_prompt(messages):
    sys_ = " ".join(m["content"] for m in messages if m["role"] == "system")
    user = " ".join(m["content"] for m in messages if m["role"] == "user")
    return (sys_ + "\n\n" if sys_ else "") + user + "\n\nHere is the story.\n\n"


def base_side(base_id, chat_text, plain_text, chat_openings, n_samples=20, T=16):
    """Base model: first-word distribution (chat text and plain framing), its own
    short openings (plain framing), and scores of both opening sets."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_id)
    m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="auto")
    m.eval()
    dev = next(m.parameters()).device
    res = {"base_id": base_id}
    for key, text in (("chat_framing", chat_text), ("plain_framing", plain_text)):
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        with torch.no_grad():
            lp = m(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1).cpu()
        v, i = lp.topk(3000)
        res[f"first_word_{key}"] = {"top": [[tok.decode([int(j)]), float(a)] for a, j in zip(v, i)],
                                    "entropy": float(-(lp.exp() * lp).sum())}
    pids = tok(plain_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    own = []
    for k in range(n_samples):
        torch.manual_seed(1000 + k)
        with torch.no_grad():
            g = m.generate(input_ids=pids, attention_mask=torch.ones_like(pids), do_sample=True,
                           temperature=1.0, top_p=None, top_k=None, max_new_tokens=T,
                           pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        own.append(tok.decode(g[0, pids.shape[-1]:], skip_special_tokens=True))
    res["base_openings"] = own
    res["base_scores_base_openings"] = _score(m, tok, pids, own, T)
    res["base_scores_chat_openings"] = _score(m, tok, pids, chat_openings, T)
    print("  [base] first word, plain framing: "
          + ", ".join(f"{w!r} {math.exp(a):.3f}" for w, a in res["first_word_plain_framing"]["top"][:8]),
          flush=True)
    print("  [base] own openings: " + " | ".join(repr(o[:60]) for o in own[:5]), flush=True)
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def _fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / max(sxx, 1e-12), sxy / max(math.sqrt(sxx * syy), 1e-12)


def compare(base, chat_lp, tok, chat_scores_base, chat_scores_chat):
    """Chat vs base: first word (matched by token text) and short openings."""
    out = {}
    for key in ("plain_framing", "chat_framing"):
        bt = {w: a for w, a in base[f"first_word_{key}"]["top"]}
        pairs, gone_n, gone_mass = [], 0, 0.0
        for w, a in bt.items():
            ids = tok(w, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                continue
            c = float(chat_lp[ids[0]])
            if a > math.log(1e-4):
                pairs.append((a, c))
            if a > math.log(1e-3) and c < -15:
                gone_n += 1
                gone_mass += math.exp(a)
        slope, corr = _fit([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else (None, None)
        out[f"first_word_{key}"] = {"tokens_compared": len(pairs), "slope_chat_on_base": slope,
                                    "correlation": corr, "base_likely_chat_impossible": gone_n,
                                    "their_base_mass": gone_mass}
    xs = [b[0] / b[1] for b in base["base_scores_base_openings"]] + \
         [b[0] / b[1] for b in base["base_scores_chat_openings"]]
    ys = [c[0] / c[1] for c in chat_scores_base] + [c[0] / c[1] for c in chat_scores_chat]
    slope, corr = _fit(xs, ys)
    nb = len(base["base_scores_base_openings"])
    out["openings"] = {
        "slope_chat_on_base_per_token": slope, "correlation": corr,
        "base_openings_mean_logp_per_token": {"base": sum(xs[:nb]) / nb, "chat": sum(ys[:nb]) / nb},
        "chat_openings_mean_logp_per_token": {"base": sum(xs[nb:]) / (len(xs) - nb),
                                              "chat": sum(ys[nb:]) / (len(ys) - nb)},
        "base_openings_chat_logp_per_token": ys[:nb]}
    for k, v in out.items():
        print(f"  [suppressed-or-gone] {k}: {json.dumps(v)[:400]}", flush=True)
    return out


def run(get_model, model_name, messages, layers, seed_for, stories_path, out_dir, n_seeds=50):
    data = json.loads(open(stories_path).read())["arms"]
    pre = PREFIX[model_name]
    arms = {k: v for k, v in data.items() if k.split(" ")[0].startswith(pre)}
    untouched = next(v for k, v in arms.items() if k.endswith("untouched"))
    res = {"model": model_name}
    prev = out_dir / "plan_checks.json"
    if prev.exists():                                # resume: keep the parts already done
        res = json.loads(prev.read_text())
        res.pop("cost", None)
    seeds = [seed_for(0, x, 0) for x in range(n_seeds)]
    # 3a. base model first, alone on the GPU
    plain = plain_prompt(messages)
    try:
        if "base" in res:
            raise StopIteration
        from transformers import AutoTokenizer
        from noiseegra.defaults import EN_MODEL_HF_IDS
        ctok = AutoTokenizer.from_pretrained(EN_MODEL_HF_IDS[model_name])
        chat_text = ctok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        res["base"] = base_side(BASE_IDS[model_name], chat_text, plain, untouched[:20])
    except StopIteration:
        pass
    except Exception as e:  # the other checks still run
        res["base_error"] = repr(e)
        print(f"  [base] FAILED: {e!r}", flush=True)
    (out_dir / "plan_checks.json").write_text(json.dumps(res, indent=1))
    egra = get_model()
    chat = egra.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = egra.tokenizer(chat, return_tensors="pt")["input_ids"].to(egra._input_device())
    hl = tuple(range(int(layers[0]), int(layers[-1]) + 1))
    # 1. spread
    if "spread" in res and "suppressed_or_gone" in res:
        lp = None
    else:
        res["spread"], lp = spread_check(egra, ids, hl, seeds, SETTINGS[model_name])
    (out_dir / "plan_checks.json").write_text(json.dumps(res, indent=1))
    # 3b. chat side of suppressed-or-gone
    if "base" in res and "suppressed_or_gone" not in res:
        sb = _score(egra.model, egra.tokenizer, ids, res["base"]["base_openings"])
        sc = _score(egra.model, egra.tokenizer, ids, untouched[:20])
        res["suppressed_or_gone"] = compare(res["base"], lp, egra.tokenizer, sb, sc)
        (out_dir / "plan_checks.json").write_text(json.dumps(res, indent=1))
    # 2. cost
    res["cost"] = cost_check(egra, ids, hl, seeds, arms)
    (out_dir / "plan_checks.json").write_text(json.dumps(res, indent=1))
    print(f"\nplan checks written to {out_dir / 'plan_checks.json'}", flush=True)
    return res
