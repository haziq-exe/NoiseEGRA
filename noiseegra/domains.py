"""Tasks beyond children's stories, each with rules the answer must not break.

The method is the same everywhere -- rule steering from contrast pairs plus the
per-story random noise sized while it is written -- so what a domain supplies is
what a task needs to be a fair test of it:

- a goal for the model, written as a system and a user message, with the rules
  listed in the request the way the story task lists its requirements;
- a check for every rule, automatic and exact, so rule-breaking is counted rather
  than judged;
- contrast pairs for every rule that has a direction to steer along: a response
  that keeps it and one that breaks it, identical elsewhere, read in the task's
  own context. A rule that only counts things ("exactly three steps") is checked
  but not steered: two continuations of one prefix cannot differ in how many
  items follow without differing in length, and the extraction reads both sides
  over the same number of positions;
- a shield, ``on_task``: answering against talking about the answer ("Sure!
  Here is..."), protected from the noise and never pushed, the role ``in_story``
  plays for stories;
- a validity check that means the same in every domain -- empty, refused,
  degenerate or looping output is invalid; a poem is not failed for being a poem;
- where the answer can be verified, a correctness check, because variety bought
  by getting the problem wrong is not variety.

Four domains. Two call for reasoning under rules and have answers that can be
checked (a number puzzle with many solutions, test cases for a function), one for
practical problem solving (a plan for a library), one is creative but not a
story (a poem).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .coherence import is_refusal

_WORD = re.compile(r"[A-Za-z0-9']+")


def words(text: str) -> List[str]:
    return _WORD.findall(text)


def plain(text: str) -> str:
    """Markdown emphasis and code fences removed, so a rule is judged on content."""
    t = text.replace("**", "").replace("__", "")
    return "\n".join(l for l in t.splitlines() if not l.strip().startswith("```"))


def lines(text: str) -> List[str]:
    return [l.strip() for l in plain(text).splitlines() if l.strip()]


def degenerate(text: str) -> Optional[str]:
    """Why a response is invalid in any domain, or None.

    Empty or near-empty, a refusal, a line repeated three or more times, or a
    tail that has collapsed onto a handful of tokens.
    """
    w = words(text)
    if len(w) < 5:
        return "too_short"
    if is_refusal(text):
        return "refusal"
    seen: Dict[str, int] = {}
    for l in lines(text):
        if len(l) >= 12:
            seen[l.lower()] = seen.get(l.lower(), 0) + 1
            if seen[l.lower()] >= 3:
                return "repeated_line"
    tail = [x.lower() for x in w[-60:]]
    if len(tail) >= 40 and len(set(tail)) / len(tail) < 0.2:
        return "loop"
    return None


@dataclass
class Rule:
    name: str
    text: str                                  # the requirement as the prompt states it
    check: Callable[[str], bool]
    pairs: List[Dict[str, str]] = field(default_factory=list)   # empty: checked, not steered


@dataclass
class Domain:
    name: str
    system: str
    task: str
    rules: List[Rule]
    closing: str
    shield_pairs: List[Dict[str, str]]
    max_new_tokens: int = 300
    max_words: int = 200
    correct: Optional[Callable[[str], Optional[bool]]] = None
    unit: str = "answer"                       # what one response is called in the prompt

    def requirements(self) -> List[str]:
        return [r.text for r in self.rules]

    def messages(self) -> List[Dict[str, str]]:
        req = "\n".join(f"- {t}" for t in self.requirements())
        user = (f"{self.task}\n\nYour {self.unit} must satisfy every one of these "
                f"requirements:\n{req}\n\n{self.closing}")
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def checks(self, text: str) -> Dict[str, bool]:
        return {r.name: bool(r.check(text)) for r in self.rules}

    def violations(self, text: str) -> int:
        return sum(not ok for ok in self.checks(text).values())

    def valid(self, text: str) -> Tuple[bool, str]:
        why = degenerate(text)
        return (why is None, why or "ok")

    @property
    def steered(self) -> List[str]:
        return [r.name for r in self.rules if r.pairs]

    def steering_pairs(self) -> Dict[str, Dict[str, list]]:
        """The contrast pairs in the shape ``load_pairs`` returns, shield included."""
        out = {r.name: {"pairs": list(r.pairs)} for r in self.rules if r.pairs}
        out["on_task"] = {"pairs": list(self.shield_pairs)}
        return out


# --------------------------------------------------------------------------- #
#  A number puzzle with many solutions                                         #
# --------------------------------------------------------------------------- #

def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(n))


SOLUTIONS = [n for n in range(1000, 10000) if n % 7 == 0 and _digit_sum(n) == 20]


def _answer_number(text: str) -> Optional[int]:
    """The number on the answer line, or else the last four-digit number given."""
    for l in reversed(lines(text)):
        m = re.match(r"^answer\s*:\s*(\d{4})\b", l, re.I)
        if m:
            return int(m.group(1))
    found = re.findall(r"\b(\d{4})\b", plain(text))
    return int(found[-1]) if found else None


def number_correct(text: str) -> Optional[bool]:
    n = _answer_number(text)
    return None if n is None else (1000 <= n <= 9999 and n % 7 == 0 and _digit_sum(n) == 20)


def _number_pairs():
    sols = SOLUTIONS[::max(1, len(SOLUTIONS) // 12)][:12]
    spell = lambda n: " + ".join(str(n))
    sum_pos = ["Its digits: {s} = 20.", "Digit sum: {s} = 20.", "Check the digits: {s} = 20.",
               "Adding the digits, {s} = 20.", "The digits give {s} = 20."]
    sum_neg = ["Its digits add up to twenty.", "Its digit sum comes to twenty.",
               "Checking the digits, they total twenty.", "Adding the digits gives twenty.",
               "The digits sum to twenty, as needed."]
    div_pos = ["Dividing: {n} ÷ 7 = {q}.", "Check: {n} ÷ 7 = {q}, no remainder.",
               "Division: {n} ÷ 7 = {q}.", "And {n} ÷ 7 = {q} exactly.", "Test: {n} ÷ 7 = {q}."]
    div_neg = ["It divides by seven evenly.", "Check: seven goes into it with no remainder.",
               "It is a multiple of seven.", "And seven divides it exactly.",
               "Test: it leaves no remainder when divided by seven."]
    ans_neg = ["So my number is {n}.", "The number I found is {n}.", "My answer is {n}.",
               "That makes {n} the number.", "{n} works."]
    long_neg = ("First, I thought about which four-digit numbers might work, and I decided to "
                "start from multiples of seven near the middle of the range, checking each "
                "one carefully to see whether its digits added up to the target of twenty.")
    digit_sum, division, answer, brief, shield = [], [], [], [], []
    for i, n in enumerate(sols):
        q, s = n // 7, spell(n)
        digit_sum.append({"prefix": f"Take {n}. ", "positive": sum_pos[i % 5].format(s=s),
                          "negative": sum_neg[i % 5]})
        division.append({"prefix": f"Take {n}. Its digits: {s} = 20. ",
                         "positive": div_pos[i % 5].format(n=n, q=q), "negative": div_neg[i % 5]})
        answer.append({"prefix": f"{s} = 20 and {n} ÷ 7 = {q}.\n", "positive": f"Answer: {n}",
                       "negative": ans_neg[i % 5].format(n=n)})
        brief.append({"prefix": f"Take {n}.", "positive": f" {s} = 20 and {n} ÷ 7 = {q}.\nAnswer: {n}",
                      "negative": " " + long_neg})
        shield.append({"prefix": "", "positive": f"Take {n}: {s} = 20, and {n} ÷ 7 = {q}.",
                       "negative": "Sure! Here is a solution to your puzzle that follows all of "
                                   "the requirements you listed."})
    return digit_sum, division, answer, brief, shield


def _number() -> Domain:
    ds, dv, an, br, sh = _number_pairs()
    return Domain(
        name="number",
        system="You are a careful problem solver. You follow every instruction exactly.",
        task=("Find a four-digit number that is divisible by 7 and whose digits add up to 20. "
              "There are many such numbers; give any one of them."),
        unit="answer",
        rules=[
            Rule("digit_sum_shown",
                 "show the digit sum written out as an addition, for example 1 + 2 + 3 + 4 = 10",
                 lambda t: bool(re.search(r"\d\s*\+\s*\d\s*\+\s*\d\s*\+\s*\d\s*=\s*\d+", plain(t))),
                 ds),
            Rule("division_shown",
                 "show the division by 7 written out, for example 1407 ÷ 7 = 201",
                 lambda t: bool(re.search(r"\b\d{4}\s*(?:÷|/|divided by)\s*7\s*=\s*\d+", plain(t),
                                          re.I)),
                 dv),
            Rule("answer_line", "the last line is exactly 'Answer:' followed by your number",
                 lambda t: bool(lines(t)) and bool(re.fullmatch(r"answer\s*:\s*\d{4}\.?", lines(t)[-1],
                                                                re.I)),
                 an),
            Rule("brief", "use at most 60 words", lambda t: len(words(t)) <= 60, br),
        ],
        closing="Write only the solution itself: no preamble or commentary.",
        shield_pairs=sh, max_new_tokens=300, max_words=120, correct=number_correct,
    )


# --------------------------------------------------------------------------- #
#  Test cases for a function                                                   #
# --------------------------------------------------------------------------- #

_CASE = re.compile(r'^is_palindrome\("((?:[^"\\]|\\.)*)"\)\s*->\s*(True|False)$')


def _unescape(s: str) -> str:
    """The Python string a quoted test input stands for; as written if it will not parse."""
    try:
        return bytes(s, "utf-8").decode("unicode_escape") if "\\" in s else s
    except (UnicodeDecodeError, ValueError):
        return s


def _cases(text: str) -> List[Tuple[str, bool]]:
    out = []
    for l in lines(text):
        m = _CASE.match(l.strip("` "))
        if m:
            out.append((_unescape(m.group(1)), m.group(2) == "True"))
    return out


def tests_correct(text: str) -> Optional[bool]:
    cs = _cases(text)
    if not cs:
        return None
    return all(expect == (s == s[::-1]) for s, expect in cs)


def _tests_pairs():
    pals = ["level", "racecar", "noon", "madam", "refer", "abba", "a", "12321", "stats", "civic",
            "Aa aA", "wow"]
    nons = ["hello", "abca", "Noon", "palindrome", "ab", "race car", "12345", "Madam", "abcd",
            "python", "level ", "no on"]
    fmt_neg = ['assert is_palindrome("{s}") == {e}', 'is_palindrome("{s}") should return {e}',
               'Input: "{s}" -> Expected: {e}', 'test("{s}", {e})', '"{s}" => {e}']
    only_neg = ["Here are five test cases for is_palindrome:", "```python",
                "Test cases for the palindrome function:", "Below are the tests you asked for:",
                "# Test cases"]
    case_format, only_cases, both, empty, shield = [], [], [], [], []
    for i in range(12):
        p, n = pals[i], nons[i]
        case_format.append({"prefix": f'is_palindrome("{pals[(i + 3) % 12]}") -> True\n',
                            "positive": f'is_palindrome("{n}") -> False',
                            "negative": fmt_neg[i % 5].format(s=n, e="False")})
        only_cases.append({"prefix": "", "positive": f'is_palindrome("{p}") -> True',
                           "negative": only_neg[i % 5]})
        both.append({"prefix": f'is_palindrome("{p}") -> True\nis_palindrome("") -> True\n',
                     "positive": f'is_palindrome("{n}") -> False',
                     "negative": f'is_palindrome("{pals[(i + 5) % 12]}") -> True'})
        empty.append({"prefix": f'is_palindrome("{p}") -> True\n',
                      "positive": 'is_palindrome("") -> True',
                      "negative": f'is_palindrome("{pals[(i + 7) % 12][0]}x") -> False'})
        shield.append({"prefix": "", "positive": f'is_palindrome("{p}") -> True',
                       "negative": "Sure! Here are five test cases for your is_palindrome function."})
    return case_format, only_cases, both, empty, shield


def _tests() -> Domain:
    cf, oc, bo, em, sh = _tests_pairs()
    return Domain(
        name="tests",
        system="You are a careful software tester. You follow every instruction exactly.",
        task=("Write five test cases for a Python function is_palindrome(s) that returns True "
              "when the string s reads the same forwards and backwards, and False otherwise. "
              "It is case-sensitive, and spaces and punctuation count as characters."),
        unit="answer",
        rules=[
            Rule("five_cases", "exactly five test cases, one per line",
                 lambda t: len(_cases(t)) == 5),
            Rule("case_format",
                 'every line has the form is_palindrome("...") -> True or is_palindrome("...") -> False',
                 lambda t: bool(_cases(t)) and all(_CASE.match(l.strip("` ")) for l in lines(t)
                                                   if "is_palindrome" in l),
                 cf),
            Rule("only_cases", "nothing but the test lines: no heading, explanation or code block",
                 lambda t: bool(lines(t)) and all(_CASE.match(l.strip("` ")) for l in lines(t))
                 and "```" not in t,
                 oc),
            Rule("both_outcomes", "at least two cases expect True and at least two expect False",
                 lambda t: sum(e for _, e in _cases(t)) >= 2 and sum(not e for _, e in _cases(t)) >= 2,
                 bo),
            Rule("empty_string", "one of the cases is the empty string",
                 lambda t: any(s == "" for s, _ in _cases(t)), em),
        ],
        closing="Write only the test lines themselves: no preamble or commentary.",
        shield_pairs=sh, max_new_tokens=300, max_words=100, correct=tests_correct,
    )


# --------------------------------------------------------------------------- #
#  A plan for a library                                                        #
# --------------------------------------------------------------------------- #

_ONLINE = re.compile(r"\b(social media|facebook|instagram|tiktok|twitter|youtube|apps?|website|"
                     r"web ?site|online|internet|e-?mail|digital|podcast|newsletter sign-?up)\b", re.I)


def _dollars(text: str) -> List[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\$\s?(\d[\d,]*(?:\.\d+)?)", text)]


def _steps(text: str) -> List[int]:
    return [int(m.group(1)) for l in lines(text) for m in [re.match(r"^(\d+)[.)]\s", l)] if m]


def _plan_pairs():
    steps = [
        ("1. Start a Saturday story hour for families.\n2. Ask the school to bring classes on Fridays.\n",
         "3. Hold a monthly book swap in the reading room.\n"),
        ("1. Open the library late on Thursdays.\n2. Set up a quiet homework corner with tutors.\n",
         "3. Host a board-game night once a month.\n"),
        ("1. Invite local authors to give short talks.\n2. Start a seed library with the garden club.\n",
         "3. Run a summer reading challenge with small prizes.\n"),
        ("1. Put a book cart at the Saturday market.\n2. Give library cards at the school fair.\n",
         "3. Start a knitting circle in the back room.\n"),
        ("1. Offer free coffee on weekday mornings.\n2. Start a job-skills workshop with the council.\n",
         "3. Lend out tools, games and puzzles.\n"),
    ]
    budget_pos = ["Total cost: ${c} for supplies and snacks.", "The whole plan costs ${c}.",
                  "Budget: ${c}, mostly for prizes and posters.", "It costs about ${c} in total.",
                  "Total: ${c} from the Friends of the Library fund."]
    budget_neg = ["It needs a modest budget from the council.", "The whole plan costs very little.",
                  "Budget: a small amount, mostly for prizes and posters.",
                  "It costs only what the library can spare.",
                  "It is paid for by the Friends of the Library fund."]
    risk_pos = ["Risk: few people may come at first.", "Risk: volunteers may drop out.",
                "Risk: the room may be too small on busy days.", "Risk: the school may say no.",
                "Risk: bad weather could keep people away."]
    risk_neg = ["One problem is that few people may come at first.",
                "Volunteers may drop out, which would be a problem.",
                "The room may be too small on busy days.", "The school might say no.",
                "Bad weather could keep people away."]
    offline_pos = ["Put up posters at the grocery store and the school.",
                   "Hand out flyers at the Saturday market.",
                   "Ask the local newspaper to print a notice.",
                   "Announce it at church and at the town meeting.",
                   "Leave bookmarks with the dates at the café."]
    offline_neg = ["Post about it on social media and the town website.",
                   "Send an email newsletter to everyone on the list.",
                   "Share it online and on the library's app.",
                   "Advertise on Facebook and Instagram.",
                   "Put the dates on the internet calendar."]
    long_neg = ("To begin with, it is important to understand why people stopped coming, which "
                "could be for many reasons, such as changes in the town, new ways of reading, busy "
                "schedules, or simply the feeling that the library no longer has anything for them.")
    costs = [120, 250, 300, 180, 450, 90, 360, 200, 275, 400]
    budget, risk, offline, brief, shield = [], [], [], [], []
    for i in range(10):
        pre, third = steps[i % 5]
        c = costs[i]
        budget.append({"prefix": pre + third, "positive": budget_pos[i % 5].format(c=c),
                       "negative": budget_neg[i % 5]})
        risk.append({"prefix": pre + third + f"Total cost: ${c}.\n", "positive": risk_pos[i % 5],
                     "negative": risk_neg[i % 5]})
        offline.append({"prefix": pre, "positive": "3. " + offline_pos[i % 5],
                        "negative": "3. " + offline_neg[i % 5]})
        brief.append({"prefix": "", "positive": pre.split("\n")[0] + "\n" + pre.split("\n")[1],
                      "negative": long_neg})
        shield.append({"prefix": "", "positive": pre.split("\n")[0],
                       "negative": "Sure! Here is a plan to bring visitors back to the library."})
    return budget, risk, offline, brief, shield


def _plan() -> Domain:
    bu, ri, of, br, sh = _plan_pairs()
    return Domain(
        name="plan",
        system="You are a practical community organiser. You follow every instruction exactly.",
        task=("The public library in a small town has lost half of its visitors in the last two "
              "years. Propose one plan to bring people back."),
        unit="plan",
        rules=[
            Rule("three_steps", "exactly three numbered steps, written 1., 2. and 3.",
                 lambda t: sorted(_steps(t)) == [1, 2, 3]),
            Rule("budget", "give the total cost in dollars, and keep it under $500",
                 lambda t: bool(_dollars(t)) and max(_dollars(t)) < 500, bu),
            Rule("risk_line", "one line that begins with 'Risk:' and names what could go wrong",
                 lambda t: any(re.match(r"^risk\s*:", l, re.I) for l in lines(t)), ri),
            Rule("offline",
                 "nothing that depends on the internet: no social media, apps, websites or anything online",
                 lambda t: not _ONLINE.search(t), of),
            Rule("brief", "use at most 120 words", lambda t: len(words(t)) <= 120, br),
        ],
        closing="Write only the plan itself: no title, preamble or commentary.",
        shield_pairs=sh, max_new_tokens=400, max_words=240,
    )


# --------------------------------------------------------------------------- #
#  A poem                                                                      #
# --------------------------------------------------------------------------- #

_BANNED = re.compile(r"\b(blue|wave|waves)\b", re.I)


def _stanzas(text: str) -> List[List[str]]:
    blocks = re.split(r"\n\s*\n", plain(text).strip())
    return [[l.strip() for l in b.splitlines() if l.strip()] for b in blocks if b.strip()]


def _poem_pairs():
    stanzas = [
        ["Salt on the wind,", "gulls on the stone,", "the tide comes in", "and leaves us alone."],
        ["Grey water breathing", "under the pier,", "the boats are sleeping,", "the night is near."],
        ["Shells in my pocket,", "sand in my shoe,", "the harbour lights", "come slowly through."],
        ["The foam is lace", "along the bay,", "it writes its name", "and slips away."],
        ["A lighthouse blinks", "across the dark,", "the rocks remember", "every spark."],
    ]
    long_lines = ["the tide comes rolling slowly in across the wide and silent grey stretch of sand",
                  "the boats are sleeping quietly in the harbour while the fishermen dream of home",
                  "the harbour lights come slowly through the mist that hangs above the water tonight",
                  "it writes its name upon the shore and then it slips away again into the deep",
                  "the rocks remember every spark that ever fell upon them from the passing ships"]
    banned_pos = ["the swell comes in", "the tide is grey", "the surf is loud", "the water turns",
                  "the foam runs white"]
    banned_neg = ["the waves come in", "the sea is blue", "the waves are loud", "the blue water turns",
                  "the wave runs white"]
    q_pos = ["Where does the tide go at night?", "Who taught the gulls to cry?",
             "What does the harbour dream?", "Why does the shore keep still?",
             "How far does the water go?"]
    q_neg = ["The tide goes somewhere at night.", "Somebody taught the gulls to cry.",
             "The harbour dreams of something.", "The shore keeps still for a reason.",
             "The water goes very far."]
    breaks, short, banned, question, shield = [], [], [], [], []
    for i in range(10):
        a, b = stanzas[i % 5], stanzas[(i + 1) % 5]
        breaks.append({"prefix": "\n".join(a) + "\n", "positive": "\n" + "\n".join(b[:2]),
                       "negative": "\n".join(b[:2])})
        short.append({"prefix": "\n".join(a[:2]) + "\n", "positive": "\n".join(a[2:]),
                      "negative": long_lines[i % 5]})
        banned.append({"prefix": "\n".join(a[:2]) + "\n", "positive": banned_pos[i % 5],
                       "negative": banned_neg[i % 5]})
        question.append({"prefix": "\n".join(a) + "\n\n", "positive": q_pos[i % 5],
                         "negative": q_neg[i % 5]})
        shield.append({"prefix": "", "positive": a[0],
                       "negative": "Sure! Here is a short poem about the sea."})
    return breaks, short, banned, question, shield


def _poem() -> Domain:
    bk, sh_, bn, qu, shield = _poem_pairs()
    return Domain(
        name="poem",
        system="You are a poet. You follow every instruction exactly.",
        task="Write a short poem about the sea.",
        unit="poem",
        rules=[
            Rule("three_stanzas", "exactly three stanzas, separated by blank lines",
                 lambda t: len(_stanzas(t)) == 3, bk),
            Rule("four_lines", "every stanza has exactly four lines",
                 lambda t: bool(_stanzas(t)) and all(len(s) == 4 for s in _stanzas(t))),
            Rule("short_lines", "no line has more than eight words",
                 lambda t: bool(lines(t)) and all(len(words(l)) <= 8 for l in lines(t)), sh_),
            Rule("banned_words", "never use the words blue, wave or waves",
                 lambda t: not _BANNED.search(t), bn),
            Rule("a_question", "at least one line ends with a question mark",
                 lambda t: any(l.rstrip().endswith("?") for l in lines(t)), qu),
        ],
        closing="Write only the poem itself: no title, preamble or commentary.",
        shield_pairs=shield, max_new_tokens=300, max_words=160,
    )


DOMAINS: Dict[str, Callable[[], Domain]] = {
    "number": _number, "tests": _tests, "plan": _plan, "poem": _poem,
}


def get(name: str) -> Domain:
    if name not in DOMAINS:
        raise KeyError(f"unknown domain {name!r}; have {sorted(DOMAINS)}")
    return DOMAINS[name]()


def domain_of(run_id: str) -> Optional[str]:
    """The domain a run id belongs to: ids are prefixed ``dom-<name>__``."""
    m = re.match(r"^dom-([a-z]+)__", run_id)
    return m.group(1) if m else None
