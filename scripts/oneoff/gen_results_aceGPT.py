import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
from noiseegra.egra_constraint_checker import EGRAConstraintChecker
from noiseegra.creativity_metrics import CreativityScorer

csv_path = Path("experiment_results/ResidNoise/AceGPT__L12-20__std0p0197__decay0.csv")
results_path = Path("experiment_results/RESULTS/AceGPT__L12-20__std0p0197__decay0.txt")
run_name = "AceGPT__L12-20__std0p0197__decay0"

# Read stories from CSV
stories = []
with open(csv_path, encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if row and row[0].strip():
            stories.append(row[0].strip())

print(f"Read {len(stories)} stories from {csv_path}")

# Generate RESULTS file
buf = io.StringIO()
with redirect_stdout(buf):
    print(f"\n\n---- {run_name} ----\n")
    CreativityScorer(stories).creativity_score(print_report=True)
    
    print(f"\n\n------------------ {run_name} CONSTRAINT ---------------------\n\n\n")
    checker = EGRAConstraintChecker()
    checker.print_report(stories)

# Write RESULTS file
results_path.write_text(buf.getvalue(), encoding="utf-8")
print(f"\nWrote RESULTS to {results_path}")
lines = buf.getvalue().split("\n")
print(f"First 30 lines:")
for i, line in enumerate(lines[:30]):
    print(line)
