"""The two-GPU shard wrapper, on a stub runner. No model, no GPU.

python tests/test_sharding.py
"""
# args through, and reports a non-zero exit. Uses a stub runner, no model.
import os, subprocess, sys, tempfile, textwrap
from pathlib import Path
ROOT = Path("/Users/haziq/Desktop/Projects/EGRA")
tmp = Path(tempfile.mkdtemp())
stub = tmp/"run_english_experiment.py"
stub.write_text(textwrap.dedent('''
    import os, sys
    print("device", os.environ.get("CUDA_VISIBLE_DEVICES"), "args", " ".join(sys.argv[1:]))
    sys.exit(3 if "--shard" in sys.argv and sys.argv[sys.argv.index("--shard")+1].startswith("1") else 0)
'''))
shard = (ROOT/"scripts"/"run_sharded.py").read_text().replace(
    'ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path(r"{tmp}")').replace(
    'RUNNER = ROOT / "scripts" / "run_english_experiment.py"', f'RUNNER = Path(r"{stub}")')
(tmp/"run_sharded.py").write_text(shard)
r = subprocess.run([sys.executable, str(tmp/"run_sharded.py"), "/out", "2", "--",
                    "--model", "Qwen3-1.7B", "--suite", "budget"],
                   capture_output=True, text=True)
print(r.stdout)
assert "[gpu0]" in r.stdout and "[gpu1]" in r.stdout, "both shards must run"
assert "device 0" in r.stdout and "device 1" in r.stdout, "each shard must be pinned"
assert "--shard 0/2" in r.stdout and "--shard 1/2" in r.stdout
assert "/out/shard0" in r.stdout and "/out/shard1" in r.stdout
assert r.returncode == 3, f"a failing shard must fail the run, got {r.returncode}"
print("shard wrapper OK: two pinned processes, distinct shards and out dirs, failure propagates")

# --- the shards are folded back into one tree before the kernel ends ---------
import json, shutil, tempfile
sys.path.insert(0, str(ROOT / "scripts"))
from run_sharded import merge  # noqa: E402

tmp2 = Path(tempfile.mkdtemp())
for i, rids in enumerate((["A", "C"], ["B", "D"])):
    d = tmp2 / f"shard{i}" / "Qwen3-1.7B"
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({
        "task": {"stories": 24}, "rms_scale": {f"k{i}": 1.5}, "entropy": {},
        "runs": {r: {"0:0": "story"} for r in rids}}))
    (d / f"{rids[0]}.csv").write_text("x")
    (d / "steering.pt").write_text("shared")
merge(tmp2)
st = json.loads((tmp2 / "Qwen3-1.7B" / "state.json").read_text())
assert sorted(st["runs"]) == ["A", "B", "C", "D"], st["runs"].keys()
assert st["rms_scale"] == {"k0": 1.5, "k1": 1.5}, "per-shard calibration must survive"
assert not list(tmp2.glob("shard*")), "the shard directories must be gone"
assert (tmp2 / "Qwen3-1.7B" / "A.csv").is_file()
assert (tmp2 / "Qwen3-1.7B" / "B.csv").is_file()
shutil.rmtree(tmp2)
print("merge OK: every condition in one state file, csvs kept, shard dirs removed")

# --- a search's candidate sets and other fields survive the merge -------------
tmp3 = Path(tempfile.mkdtemp())
for i, rid in enumerate(["S1", "S2"]):
    d = tmp3 / f"shard{i}" / "Qwen3-1.7B"
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({
        "task": {"stories": 2}, "rms_scale": {}, "entropy": {},
        "prompts": ["p0", "p1"],
        "runs": {rid: {f"0:{i}": "chosen"}, "SHARED": {f"0:{i}": "x"}},
        "candidates": {rid: {f"0:{i}": [{"path": 0, "text": "a", "score": -1.0}]},
                       "SHARED": {f"0:{i}": [{"path": 1, "text": "b", "score": -2.0}]}}}))
merge(tmp3)
st = json.loads((tmp3 / "Qwen3-1.7B" / "state.json").read_text())
assert st.get("prompts") == ["p0", "p1"], "fields beyond the runs must survive"
assert sorted(st["candidates"]) == ["S1", "S2", "SHARED"], st.get("candidates")
assert sorted(st["candidates"]["SHARED"]) == ["0:0", "0:1"], "one run's sets from both shards"
assert st["candidates"]["S1"]["0:0"][0]["text"] == "a"
shutil.rmtree(tmp3)
print("merge OK: candidate sets from both shards kept, prompts kept")
