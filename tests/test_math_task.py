"""GSM8K answer handling.

    python tests/test_math_task.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noiseegra.math_task import build_messages, extract_answer, gold_answer, is_correct  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


check("a reference solution's answer is the number after ####",
      gold_answer("She has 3 + 4 = <<3+4=7>>7 apples.\\n#### 7") == 7.0)
check("with thousands separators", gold_answer("... #### 1,250") == 1250.0)
check("the model's answer is the last '####' number",
      extract_answer("First #### 5 was wrong.\\nSo the answer is:\\n#### 12") == 12.0)
check("or, without one, the last number", extract_answer("So she pays $18.50 in total.") == 18.5)
check("a dollar sign and a full stop are ignored", extract_answer("#### $40.") == 40.0)
check("no number is no answer", extract_answer("I cannot tell.") is None)
check("a boxed answer is taken before a later stray number",
      extract_answer("### Final Answer: $$ \\boxed{16} $$ (over 2 days)") == 16.0)
check("correct when equal as numbers", is_correct("#### 7.0", 7.0) and not is_correct("#### 8", 7.0))
m = build_messages("Tom has 3 apples. How many?")
check("the prompt asks for the #### line", m[1]["content"].endswith("#### <number>") and m[0]["role"] == "system")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
