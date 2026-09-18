#!/usr/bin/env python
"""The contrast-pair files must not smuggle a register difference into a direction.

A steering direction is the average difference between the activations of a
positive and a negative continuation. Whatever the two sides differ in, the
direction encodes. If the positives are systematically shorter-sentenced than
the negatives, the direction means "write shorter sentences" as much as it means
the property it is named after -- and applying it to a task that asks for real
sentences flattens the prose.

That is not hypothetical. On the middle-school task the constraint push dropped
the share of stories reaching a grade-3 reading level from 57% to 4%, and the
first draft of the middle-school pair file reintroduced the same fault in its
dialogue pairs (positives at 11.6 words a sentence against negatives at 14.4)
before this test was written.

These checks run offline in under a second and need no model.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.constraint_metrics_en import flesch_kincaid_grade  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "noiseegra" / "data"
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE = re.compile(r"[^.!?]+[.!?]")

# `mature_register` is the one direction whose whole purpose is to contrast
# developed sentences with clipped ones, so its two sides are meant to differ in
# sentence length. Every other direction must not.
LENGTH_IS_THE_POINT = {"mature_register", "terse", "simple_syntax", "simple_register",
                       "no_heading", "story_format"}

# These four differ in sentence structure at a matched word count, so the
# word-count rule still applies to them. `no_heading` is the one direction where
# the word count itself is the property: its two sides share a body word for
# word and the negative adds a heading above it. Matching the totals would mean
# shortening the body under the heading, which would make the direction "write
# less" as well as "write no heading" -- the confound this file exists to catch.
WORD_COUNT_IS_THE_POINT = {"no_heading", "story_format"}


def words(text: str):
    return WORD.findall(text)


def words_per_sentence(text: str) -> float:
    sentences = SENTENCE.findall(text) or [text]
    return len(words(text)) / len(sentences)


def check_file(path: Path, *, min_words_per_sentence: float) -> None:
    pairs = json.loads(path.read_text(encoding="utf-8"))["constraints"]
    print(f"\n{path.name}: {len(pairs)} directions")

    for name, entry in sorted(pairs.items()):
        items = entry["pairs"]
        assert items, f"{name}: no pairs"

        for i, item in enumerate(items):
            for side in ("positive", "negative"):
                assert item.get(side, "").strip(), f"{name}[{i}]: empty {side}"
            assert item["positive"] != item["negative"], f"{name}[{i}]: sides identical"

        gaps = [len(words(it["positive"])) - len(words(it["negative"])) for it in items]
        mean_gap = sum(gaps) / len(gaps)
        worst = max(abs(g) for g in gaps)

        pos_wps = sum(words_per_sentence(it["positive"]) for it in items) / len(items)
        neg_wps = sum(words_per_sentence(it["negative"]) for it in items) / len(items)

        # The difference vector must not encode "one side is longer".
        if name not in WORD_COUNT_IS_THE_POINT:
            assert abs(mean_gap) <= 2.0, (
                f"{name}: positives average {mean_gap:+.2f} words against their "
                f"negatives; the direction would carry length as well as the property"
            )
            assert worst <= 4, f"{name}: one pair differs by {worst} words"

        if name not in LENGTH_IS_THE_POINT:
            assert abs(pos_wps - neg_wps) <= 2.5, (
                f"{name}: positives run at {pos_wps:.1f} words a sentence against "
                f"the negatives' {neg_wps:.1f}; pushing this direction would change "
                f"sentence length, which is what collapsed the reading floor"
            )
            # Both sides have to be written in the register the task asks for.
            for side, value in (("positives", pos_wps), ("negatives", neg_wps)):
                assert value >= min_words_per_sentence, (
                    f"{name}: {side} run at {value:.1f} words a sentence, below the "
                    f"{min_words_per_sentence} this file is meant to be written at"
                )

        print(f"  {name:>16}  word gap {mean_gap:+5.2f} (worst {worst})  "
              f"words a sentence {pos_wps:5.1f} / {neg_wps:5.1f}")


def test_middle_pairs_are_written_at_middle_school_length() -> None:
    check_file(DATA / "steering_pairs_en_middle.json", min_words_per_sentence=8.0)


def test_children_pairs_are_internally_balanced() -> None:
    # The children's file is deliberately written in a small register, so it is
    # held only to the balance requirement, not to a sentence-length floor.
    check_file(DATA / "steering_pairs_en.json", min_words_per_sentence=0.0)


def test_middle_file_covers_every_steered_direction() -> None:
    from noiseegra.defaults import EN_MIDDLE_REGISTER_STEER_VECTORS

    have = set(json.loads(
        (DATA / "steering_pairs_en_middle.json").read_text(encoding="utf-8")
    )["constraints"])
    missing = [n for n in EN_MIDDLE_REGISTER_STEER_VECTORS if n not in have]
    assert not missing, f"no contrast pairs for {missing}"
    print(f"\nall {len(EN_MIDDLE_REGISTER_STEER_VECTORS)} steered directions have pairs")


def test_reading_grade_separates_the_register_direction() -> None:
    """The register direction's two sides must actually differ in reading grade.

    Without this the direction could be named `mature_register` and contrast
    nothing the reading floor can see.
    """
    pairs = json.loads(
        (DATA / "steering_pairs_en_middle.json").read_text(encoding="utf-8")
    )["constraints"]["mature_register"]["pairs"]

    def grade(text: str) -> float:
        sentences = SENTENCE.findall(text) or [text]
        return flesch_kincaid_grade(words(text), len(sentences))

    pos = sum(grade(it["positive"]) for it in pairs) / len(pairs)
    neg = sum(grade(it["negative"]) for it in pairs) / len(pairs)
    assert pos - neg >= 4.0, (
        f"the register direction's positives read at grade {pos:.1f} and its "
        f"negatives at {neg:.1f}; that is too small a gap to steer the floor"
    )
    print(f"register direction: positives grade {pos:.1f}, negatives grade {neg:.1f}")


if __name__ == "__main__":
    test_middle_pairs_are_written_at_middle_school_length()
    test_children_pairs_are_internally_balanced()
    test_middle_file_covers_every_steered_direction()
    test_reading_grade_separates_the_register_direction()
    print("\nok")
