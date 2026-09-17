"""Resume across shards: seeding and merge must not lose history.

A finished sharded run is merged into one tree and that is what the checkpoint
carries back, but the shards read and write their own subdirectories. Without
seeding they start from nothing and regenerate everything -- two hours of GPU on
an 8B run before it was noticed. And rebuilding the merged file from the shard
states alone discarded the restored history, so a resumed run came back holding
only what it generated that time.

python tests/test_shard_resume.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sharded import merge, seed_shards
fails = []
def check(n, c, d=""):
    print(f"{'[ok] ' if c else '[FAIL] '}{n}" + (f"  {d}" if d else ""))
    if not c: fails.append(n)

tmp = Path(tempfile.mkdtemp())
# a finished run: merged tree with two conditions
md = tmp / "Qwen3-8B"; md.mkdir(parents=True)
(md / "state.json").write_text(json.dumps({"task": {"t": 1},
    "runs": {"A": {"0:0": "a0", "0:1": "a1"}, "B": {"0:0": "b0"}},
    "rms_scale": {"k": 1.0}}))
(md / "steering.pt").write_bytes(b"weights")

seed_shards(tmp, 2)
s0 = json.loads((tmp / "shard0" / "Qwen3-8B" / "state.json").read_text())
check("each shard is seeded with the merged history",
      len(s0["runs"]["A"]) == 2 and s0["runs"]["B"]["0:0"] == "b0")
check("cached vectors are seeded too, so they are not re-extracted",
      (tmp / "shard1" / "Qwen3-8B" / "steering.pt").is_file())

# shard1 now generates a NEW condition; shard0 adds nothing
s1p = tmp / "shard1" / "Qwen3-8B" / "state.json"
s1 = json.loads(s1p.read_text()); s1["runs"]["C"] = {"0:0": "c0"}
s1p.write_text(json.dumps(s1))
merge(tmp)
after = json.loads((md / "state.json").read_text())
check("the earlier run's conditions survive the merge",
      len(after["runs"]["A"]) == 2 and "B" in after["runs"], sorted(after["runs"]))
check("the new condition is added", after["runs"]["C"]["0:0"] == "c0")
check("the task record survives", after.get("task") == {"t": 1})

# and a first run, with no merged tree, still works
tmp2 = Path(tempfile.mkdtemp())
seed_shards(tmp2, 2)
d = tmp2 / "shard0" / "M"; d.mkdir(parents=True)
(d / "state.json").write_text(json.dumps({"runs": {"X": {"0:0": "x"}}}))
merge(tmp2)
check("a first run with no history still merges",
      json.loads((tmp2 / "M" / "state.json").read_text())["runs"]["X"]["0:0"] == "x")
shutil.rmtree(tmp); shutil.rmtree(tmp2)
print()
if fails: print(f"{len(fails)} FAILED: {fails}"); raise SystemExit(1)
print("resume tests passed")
