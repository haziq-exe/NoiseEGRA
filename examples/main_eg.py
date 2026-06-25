from pathlib import Path

from noiseegra.models.Jais import Jais

out_dir = Path(__file__).resolve().parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)

model = Jais()
model.zero_shot(
    output_file=str(out_dir / "example.csv"),
    num_stories=1,
    max_new_tokens=500,
    do_sample=True,
    include_sys=True,
)
