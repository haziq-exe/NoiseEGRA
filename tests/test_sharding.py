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
