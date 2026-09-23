"""The tasks beyond stories: every rule, answer check and contrast pair.

    python tests/test_domains.py

For each domain, one answer written to keep every rule and variants that break
exactly one, so each check is shown to catch its own rule and nothing else; the
answer checks on right and wrong answers; the validity check on degenerate
output; and the contrast pairs well formed -- both sides present and different,
the shield included, and none for a rule that only counts.
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noiseegra import domains as D  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


def only_breaks(dom, text, rule):
    got = dom.checks(text)
    broken = [r for r, ok in got.items() if not ok]
    return broken == [rule], broken


GOOD = {
    "number": "Try 2567.\n2 + 5 + 6 + 7 = 20.\n2567 ÷ 7 = 366.71, so not that; try 3476.\n"
              "3 + 4 + 7 + 6 = 20 and 3476 ÷ 7 = 496.57... Try 5663: 5 + 6 + 6 + 3 = 20, "
              "5663 ÷ 7 = 809.\nAnswer: 5663",
    "tests": 'is_palindrome("racecar") -> True\nis_palindrome("") -> True\n'
             'is_palindrome("hello") -> False\nis_palindrome("Noon") -> False\n'
             'is_palindrome("a b a") -> True',
    "plan": "1. Start a Saturday story hour for families.\n2. Lend board games and puzzles.\n"
            "3. Hold a monthly book swap.\nTotal cost: $300 for supplies.\n"
            "Risk: few people may come at first.",
    "poem": "Salt on the wind,\ngulls on the stone,\nthe tide comes in\nand leaves us alone.\n\n"
            "Grey water breathing\nunder the pier,\nthe boats are sleeping,\nthe night is near.\n\n"
            "Where does it go\nwhen the moon is thin?\nThe harbour waits\nto let it in?",
}
BREAK = {
    "number": {
        "digit_sum_shown": GOOD["number"].replace("5 + 6 + 6 + 3 = 20", "the digits make twenty")
                           .replace("2 + 5 + 6 + 7 = 20", "digits fine").replace("3 + 4 + 7 + 6 = 20", "ok"),
        "division_shown": GOOD["number"].replace("5663 ÷ 7 = 809", "it divides by seven")
                          .replace("2567 ÷ 7 = 366.71", "no").replace("3476 ÷ 7 = 496.57...", "no."),
        "answer_line": GOOD["number"].replace("Answer: 5663", "So the number is 5663."),
        "brief": GOOD["number"].replace("\nAnswer:", "\n" + " ".join(["checking"] * 40) + "\nAnswer:"),
    },
    "tests": {
        "five_cases": GOOD["tests"] + '\nis_palindrome("abba") -> True',
        "case_format": GOOD["tests"].replace('is_palindrome("hello") -> False',
                                             'assert is_palindrome("hello") == False'),
        "only_cases": "Here are the tests:\n" + GOOD["tests"],
        "both_outcomes": GOOD["tests"].replace('("Noon") -> False', '("noon") -> True'),
        "empty_string": GOOD["tests"].replace('("") -> True', '("x") -> True'),
    },
    "plan": {
        "three_steps": GOOD["plan"].replace("Risk:", "4. Add a café corner.\nRisk:"),
        "budget": GOOD["plan"].replace("$300", "$800"),
        "risk_line": GOOD["plan"].replace("Risk: few", "The risk is that few"),
        "offline": GOOD["plan"].replace("board games", "board games and a website"),
        "brief": GOOD["plan"] + "\n" + " ".join(["more"] * 100),
    },
    "poem": {
        "three_stanzas": GOOD["poem"] + "\n\nOne more line\nand then\nanother\nto end.",
        "four_lines": GOOD["poem"].replace("and leaves us alone.\n", "and leaves us alone.\nfor now.\n"),
        "short_lines": GOOD["poem"].replace("under the pier,", "under the long wooden pier where the old men fish,"),
        "banned_words": GOOD["poem"].replace("Grey water", "Blue water"),
        "a_question": GOOD["poem"].replace("?", "."),
    },
}

COUPLED = {("tests", "case_format")}

for name in D.DOMAINS:
    dom = D.get(name)
    print(f"== {name} ==")
    got = dom.checks(GOOD[name])
    check(f"{name}: the good answer keeps every rule", all(got.values()),
          str([r for r, ok in got.items() if not ok]))
    for rule, text in BREAK[name].items():
        ok, broken = only_breaks(dom, text, rule)
        if (name, rule) in COUPLED:
            # A line in the wrong form is not a test case at all, so it also
            # changes the count and the balance of outcomes.
            check(f"{name}: breaking {rule} is caught", rule in broken, str(broken))
        else:
            check(f"{name}: breaking {rule} breaks it alone", ok, str(broken))
    check(f"{name}: a good answer is valid", dom.valid(GOOD[name])[0])
    check(f"{name}: an empty one is not", not dom.valid("")[0])
    check(f"{name}: nor a line said three times", not dom.valid("\n".join(["the same line again here"] * 4))[0])
    check(f"{name}: nor a refusal", not dom.valid("I'm sorry, but I can't help with that request.")[0])
    msg = dom.messages()
    check(f"{name}: the request lists every rule",
          all(t in msg[1]["content"] for t in dom.requirements()) and msg[0]["role"] == "system")
    pairs = dom.steering_pairs()
    check(f"{name}: every steered rule and the shield have pairs, both sides present and different",
          set(pairs) == set(dom.steered) | {"on_task"}
          and all(p["positive"] and p["negative"] and p["positive"] != p["negative"]
                  for v in pairs.values() for p in v["pairs"])
          and all(len(v["pairs"]) >= 8 for v in pairs.values()),
          f"{ {k: len(v['pairs']) for k, v in pairs.items()} }")

print("== answers ==")
check("a right number is right", D.number_correct("Answer: 5663") is True)
check("a wrong one is wrong", D.number_correct("Answer: 5664") is False)
check("with no answer line, the last number given counts", D.number_correct("I pick 2963 then 5663.") is True)
check("with no number at all there is no verdict", D.number_correct("I cannot find one.") is None)
check("every listed solution is one", all(n % 7 == 0 and sum(map(int, str(n))) == 20 for n in D.SOLUTIONS)
      and len(D.SOLUTIONS) > 50, f"{len(D.SOLUTIONS)} solutions")
check("right test cases are right", D.tests_correct(GOOD["tests"]) is True)
check("a wrong expectation is caught", D.tests_correct('is_palindrome("abc") -> True') is False)
check("an escaped quote is read as one", D.tests_correct('is_palindrome("a\\"a") -> True') is True)
check("no parsable case, no verdict", D.tests_correct("assert is_palindrome('x')") is None)
check("run ids carry their domain", D.domain_of("dom-poem__Qwen3-1.7B__BASELINE") == "poem"
      and D.domain_of("Qwen3-1.7B__BASELINE") is None)

print("== the runner builds the headline method for each task ==")
sys.path.insert(0, str(ROOT / "scripts"))
import types  # noqa: E402
import run_domains as RD  # noqa: E402
RD.set_run_defaults()
args = types.SimpleNamespace(model="Qwen3-1.7B", steer_budget=2.5, baseline_temperature=1.8)
for name in D.DOMAINS:
    dom = D.get(name)
    layers = [2, 3]
    specs = RD.specs_for(dom, RD.stand_in_vectors(dom, layers), layers, 1.0, args)
    arms = [a for a, _, _ in specs]
    rid = {a: r for a, r, _ in specs}
    plan = dict((a, sp) for a, _, sp in specs)["method"].steering_plan
    check(f"{name}: untouched, top-p and the method, with ids of their own",
          arms == ["untouched", "topp", "method"] and len(set(rid.values())) == 3
          and all(r.startswith(f"dom-{name}__") for r in rid.values()))
    check(f"{name}: the method is the headline one -- sized while writing, prompt 1.5x, "
          "steering 2.5 on the task's own rules, the shield kept clear",
          plan.offset_online == 1.0 and plan.offset_prefill_gain == 1.5
          and plan.steer_budget == 2.5 and [sp.name for sp in plan.specs] == dom.steered
          and "__online1__opg1p5__cn2__tail8__prefill" in rid["method"]
          and "__bud2p5__" in rid["method"] and "__rr64__" in rid["method"])
    check(f"{name}: top-p at 1.8", "__temp1p8__topp0p95" in rid["topp"])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
