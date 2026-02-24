from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class ConstraintResult:
    constraint_id: str
    description: str
    status: str  # "checked" | "not_checked"
    passed: Optional[bool]
    details: str


class EGRAConstraintChecker:
    """
    High-confidence EGRA constraint checker.

    Design principles:
    - Only constraints that can be checked with strong, explicit correctness are enforced.
    - Subjective/semantic constraints are marked "not_checked" with a reason.
    - No keyword-lexicon heuristics are used for semantic judgments.

    Name-based checks use one of:
    1) `name_extractor`: custom callable(text) -> list[str]
    2) `spacy_model`: spaCy model with PERSON/PER NER labels

    If neither is provided, name-based constraints are not checked.
    """

    CONSTRAINTS: Dict[str, str] = {
        "C01": "Narrative structure includes intro, middle dilemma, and ending with resolution.",
        "C02": "Story must not exceed 60 words.",
        "C03": "One or two child-context Arabic names, common but not textbook-heavy.",
        "C04": "Child-appropriate and positive familiar content.",
        "C05": "Contains short-story elements: character, context, beginning, obstacle, solution.",
        "C06": "Gender-balanced: includes both a boy and a girl.",
        "C07": "Avoids gender/religion/other stereotypes.",
        "C08": "No references to known stories or myths.",
        "C09": "Uses present tense.",
        "C10": "Vocabulary suitable for children and local context.",
        "C11": "First sentence is very easy.",
        "C12": "Varied but non-literary/non-overly-complex sentence structure.",
        "C13": "Supports literal and inferential comprehension questions.",
        "C14": "Uses exactly one proper noun (common name).",
        "C15": "Avoids ambiguous words with multiple possible readings/spellings.",
        "C16": "Not a weakly connected list of sentences.",
        "C17": "Avoids character names common in school textbooks.",
        "C18": "Contains one or two characters only.",
        "C19": "Includes some moderately complex vocabulary and sentence structures.",
    }

    NOT_CHECKED_REASONS: Dict[str, str] = {
        "C01": "Not reliably decidable from text alone with strong correctness.",
        "C03": "Cannot strongly verify 'common in child context' and 'not textbook-heavy' without external gold resources.",
        "C04": "Child-appropriateness and positive affect are semantic judgments, not strongly decidable by deterministic rules.",
        "C05": "Narrative elements cannot be strongly verified without deep semantic annotation.",
        "C07": "Stereotype detection is semantic/contextual and not strongly decidable via deterministic checks.",
        "C08": "Detecting originality vs prior stories/myths requires external corpus matching.",
        "C10": "Age/region suitability needs curriculum-level lexical standards and locale metadata.",
        "C12": "Style complexity/literariness is subjective without validated stylistic scoring models.",
        "C13": "Question-answer affordance is a downstream pedagogy property, not strictly text-decidable.",
        "C15": "Lexical ambiguity cannot be strongly decided without contextual disambiguation and orthographic standards.",
        "C16": "Narrative cohesion quality is semantic/discourse-level and not strongly decidable by deterministic rules.",
        "C19": "Required level of complexity is subjective without a validated complexity rubric.",
    }

    # C17 is repurposed per request into a strict cross-story uniqueness check.
    C17_REINTERPRETED_DESCRIPTION = (
        "Name uniqueness across generated stories: a character name must not be reused in multiple stories."
    )

    def __init__(
        self,
        name_extractor: Optional[Callable[[str], Sequence[str]]] = None,
        spacy_model: Optional[str] = None,
    ):
        if name_extractor is not None and not callable(name_extractor):
            raise TypeError("name_extractor must be callable(text) -> Sequence[str].")

        self._name_extractor = name_extractor
        self._spacy_nlp = None

        if spacy_model is not None:
            try:
                import spacy
            except Exception as exc:
                raise RuntimeError("spaCy is required when spacy_model is provided.") from exc
            self._spacy_nlp = spacy.load(spacy_model)

    def _normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"[\u064B-\u0652]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenize_words(self, text: str) -> List[str]:
        # Unicode-word tokenizer: Arabic/Latin words and digits.
        return re.findall(r"[\u0621-\u064A]+|[A-Za-z]+|\d+", text)

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"[\.!\؟\?\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _count_strong_tense_signals(self, words: Sequence[str]) -> Dict[str, int]:
        """
        Conservative morphology-only tense evidence.
        We only count high-confidence conjugation patterns, avoiding lexical lists.
        """
        arabic_words = [w for w in words if re.fullmatch(r"[\u0621-\u064A]+", w)]

        # Strong present forms (prefix conjugation): يبدأ بأحد أحرف المضارعة.
        present_re = re.compile(r"^[أنيت][\u0621-\u064A]{2,}(?:ون|ين|ان|وا|ن)?$")

        # Strong past forms with explicit suffix conjugation.
        past_suffix_re = re.compile(r"^[\u0621-\u064A]{2,}(?:ت|نا|تم|تن|وا)$")

        present = 0
        past = 0
        for w in arabic_words:
            if present_re.match(w):
                present += 1
            if past_suffix_re.match(w):
                past += 1

        return {"present": present, "past": past}

    def _evaluate_c09_present_tense(self, words: Sequence[str]) -> ConstraintResult:
        signals = self._count_strong_tense_signals(words)
        present = signals["present"]
        past = signals["past"]

        # Conservative decision:
        # - fail if past is clearly dominant
        # - pass if present is clearly dominant
        # - otherwise not checked (ambiguous morphology)
        if past >= 2 and past > (2 * present):
            return self._result_checked(
                "C09",
                passed=False,
                details=f"Strong past dominance detected (past={past}, present={present}).",
            )

        if present >= 2 and present > (2 * past):
            return self._result_checked(
                "C09",
                passed=True,
                details=f"Strong present dominance detected (present={present}, past={past}).",
            )

        return self._result_not_checked(
            "C09",
            f"Ambiguous tense morphology (present={present}, past={past}); no strong decision.",
        )

    def _evaluate_c06_gender_balance(self, words: Sequence[str]) -> ConstraintResult:
        """
        Conservative gender-balance check from explicit pronoun+verb agreement only.
        This avoids content-word lexicons and ignores noun grammatical gender.
        """
        arabic_words = [w for w in words if re.fullmatch(r"[\u0621-\u064A]+", w)]
        masculine = 0
        feminine = 0

        for i in range(len(arabic_words) - 1):
            subj = arabic_words[i]
            verb = arabic_words[i + 1]

            if subj == "هو" and re.match(r"^ي[\u0621-\u064A]{2,}(?:ون|ان|وا|ن)?$", verb):
                masculine += 1
            elif subj == "هم" and re.match(r"^ي[\u0621-\u064A]{2,}(?:ون|وا)$", verb):
                masculine += 1
            elif subj == "هي" and re.match(r"^ت[\u0621-\u064A]{2,}(?:ين|ان|ن)?$", verb):
                feminine += 1
            elif subj == "هن" and re.match(r"^[يت][\u0621-\u064A]{2,}ن$", verb):
                feminine += 1

        if masculine == 0 and feminine == 0:
            return self._result_not_checked(
                "C06",
                "No strong explicit gendered subject-verb signals were found.",
            )

        if masculine >= 1 and feminine >= 1:
            return self._result_checked(
                "C06",
                passed=True,
                details=f"Balanced explicit signals detected (masculine={masculine}, feminine={feminine}).",
            )

        if masculine >= 2 and feminine == 0:
            return self._result_checked(
                "C06",
                passed=False,
                details=f"Strongly masculine-skewed explicit signals (masculine={masculine}, feminine={feminine}).",
            )

        if feminine >= 2 and masculine == 0:
            return self._result_checked(
                "C06",
                passed=False,
                details=f"Strongly feminine-skewed explicit signals (masculine={masculine}, feminine={feminine}).",
            )

        return self._result_not_checked(
            "C06",
            f"Weak/insufficient explicit gendered signals (masculine={masculine}, feminine={feminine}).",
        )

    def _evaluate_c11_first_sentence_easy(self, text: str) -> ConstraintResult:
        """
        Conservative first-sentence simplicity check:
        - short sentence length
        - no strong compound/complex sentence markers
        - at most one strong finite-verb signal
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return self._result_not_checked("C11", "No sentence found.")

        first = sentences[0]
        words = self._tokenize_words(first)
        if not words:
            return self._result_not_checked("C11", "First sentence has no detectable words.")

        compound_connectors = {"ثم", "لكن", "لان", "لأن", "عندما", "اذا", "إذا", "بينما", "رغم"}
        connector_count = sum(1 for w in words if w in compound_connectors)
        punctuation_compound = bool(re.search(r"[،;:]", first))

        tense_signals = self._count_strong_tense_signals(words)
        finite_verb_proxy = tense_signals["present"] + tense_signals["past"]

        compound_evidence = int(connector_count > 0) + int(punctuation_compound) + int(finite_verb_proxy >= 2)
        length = len(words)

        if length <= 8 and compound_evidence == 0 and finite_verb_proxy <= 1:
            return self._result_checked(
                "C11",
                passed=True,
                details=f"Simple first sentence detected (tokens={length}, finite_verb_proxy={finite_verb_proxy}).",
            )

        if length > 14 or compound_evidence >= 2 or (length >= 12 and compound_evidence >= 1):
            return self._result_checked(
                "C11",
                passed=False,
                details=(
                    "First sentence appears not easy "
                    f"(tokens={length}, compound_evidence={compound_evidence}, finite_verb_proxy={finite_verb_proxy})."
                ),
            )

        return self._result_not_checked(
            "C11",
            (
                "First sentence complexity is ambiguous "
                f"(tokens={length}, compound_evidence={compound_evidence}, finite_verb_proxy={finite_verb_proxy})."
            ),
        )

    def _normalize_name(self, name: str) -> str:
        name = self._normalize(name)
        return re.sub(r"\s+", " ", name)

    def _extract_person_names(self, text: str) -> Dict[str, object]:
        if self._name_extractor is not None:
            raw = self._name_extractor(text)
            if raw is None:
                raw = []
            names = []
            for item in raw:
                if isinstance(item, str):
                    n = self._normalize_name(item)
                    if n:
                        names.append(n)
            unique = sorted(set(names))
            return {"available": True, "method": "custom", "names": unique}

        if self._spacy_nlp is not None:
            doc = self._spacy_nlp(text)
            names = []
            for ent in doc.ents:
                if ent.label_.upper() in {"PERSON", "PER"}:
                    n = self._normalize_name(ent.text)
                    if n:
                        names.append(n)
            unique = sorted(set(names))
            return {"available": True, "method": "spacy", "names": unique}

        return {
            "available": False,
            "method": None,
            "names": [],
            "reason": "No name extractor configured. Provide `name_extractor` or `spacy_model`.",
        }

    def _result_checked(self, constraint_id: str, passed: bool, details: str) -> ConstraintResult:
        description = self.C17_REINTERPRETED_DESCRIPTION if constraint_id == "C17" else self.CONSTRAINTS[constraint_id]
        return ConstraintResult(
            constraint_id=constraint_id,
            description=description,
            status="checked",
            passed=bool(passed),
            details=details,
        )

    def _result_not_checked(self, constraint_id: str, reason: str) -> ConstraintResult:
        description = self.C17_REINTERPRETED_DESCRIPTION if constraint_id == "C17" else self.CONSTRAINTS[constraint_id]
        return ConstraintResult(
            constraint_id=constraint_id,
            description=description,
            status="not_checked",
            passed=None,
            details=reason,
        )

    def _evaluate_single_story(
        self,
        text: str,
        extracted_names: Sequence[str],
        names_available: bool,
        name_unavailable_reason: Optional[str],
        repeated_names_global: Optional[set],
    ) -> List[ConstraintResult]:
        results: List[ConstraintResult] = []
        words = self._tokenize_words(text)

        # C02: deterministic and strongly checkable.
        results.append(
            self._result_checked(
                "C02",
                passed=(len(words) <= 60),
                details=f"word_count={len(words)} (must be <= 60)",
            )
        )

        # C09: checked only when tense evidence is strongly dominant.
        results.append(self._evaluate_c09_present_tense(words))
        # C06: checked only with strong explicit gendered subject-verb evidence.
        results.append(self._evaluate_c06_gender_balance(words))
        # C11: checked only with strong simple/compound evidence.
        results.append(self._evaluate_c11_first_sentence_easy(text))

        # C14 + C18: require reliable PERSON extraction.
        if names_available:
            unique_names = sorted(set(extracted_names))

            results.append(
                self._result_checked(
                    "C14",
                    passed=(len(unique_names) == 1),
                    details=f"detected_unique_person_names={unique_names}",
                )
            )

            results.append(
                self._result_checked(
                    "C18",
                    passed=(1 <= len(unique_names) <= 2),
                    details=f"detected_unique_person_names={unique_names}",
                )
            )

            if repeated_names_global is not None:
                reused = sorted([n for n in unique_names if n in repeated_names_global])
                results.append(
                    self._result_checked(
                        "C17",
                        passed=(len(reused) == 0),
                        details=f"reused_names_across_stories={reused}",
                    )
                )
            else:
                results.append(
                    self._result_not_checked(
                        "C17",
                        "Cross-story uniqueness requires evaluating a full story set.",
                    )
                )
        else:
            reason = name_unavailable_reason or "Name extraction unavailable."
            results.append(self._result_not_checked("C14", reason))
            results.append(self._result_not_checked("C18", reason))
            results.append(self._result_not_checked("C17", reason))

        # Remaining constraints are intentionally not checked due non-strong verifiability.
        for cid in [
            "C01", "C03", "C04", "C05", "C07", "C08", "C10", "C12", "C13", "C15", "C16", "C19",
        ]:
            results.append(self._result_not_checked(cid, self.NOT_CHECKED_REASONS[cid]))

        # Stable ordering by constraint id.
        results.sort(key=lambda r: int(r.constraint_id[1:]))
        return results

    def evaluate_story(self, story: str) -> Dict[str, object]:
        if not isinstance(story, str):
            raise TypeError("story must be a string.")

        normalized = self._normalize(story)
        names_info = self._extract_person_names(normalized)
        results = self._evaluate_single_story(
            text=normalized,
            extracted_names=names_info["names"],
            names_available=bool(names_info["available"]),
            name_unavailable_reason=names_info.get("reason"),
            repeated_names_global=None,
        )

        return self._build_story_output(normalized, names_info, results)

    def _build_story_output(self, normalized_story: str, names_info: Dict[str, object], results: Sequence[ConstraintResult]) -> Dict[str, object]:
        checked = [r for r in results if r.status == "checked"]
        violations = [r for r in checked if r.passed is False]
        not_checked = [r for r in results if r.status == "not_checked"]

        return {
            "story_text": normalized_story,
            "word_count": len(self._tokenize_words(normalized_story)),
            "detected_person_names": list(names_info.get("names", [])),
            "name_extraction": {
                "available": bool(names_info.get("available", False)),
                "method": names_info.get("method"),
                "reason": names_info.get("reason"),
            },
            "violations_count": len(violations),
            "checked_constraints_count": len(checked),
            "not_checked_constraints_count": len(not_checked),
            "violated_constraint_ids": [r.constraint_id for r in violations],
            "not_checked_constraint_ids": [r.constraint_id for r in not_checked],
            "constraint_results": [
                {
                    "constraint_id": r.constraint_id,
                    "description": r.description,
                    "status": r.status,
                    "passed": r.passed,
                    "details": r.details,
                }
                for r in results
            ],
        }

    def evaluate_stories(self, stories: Sequence[str]) -> Dict[str, object]:
        if not isinstance(stories, (list, tuple)) or len(stories) == 0:
            raise ValueError("stories must be a non-empty list/tuple of strings.")

        normalized_stories: List[str] = []
        names_info_per_story: List[Dict[str, object]] = []

        for i, story in enumerate(stories):
            if not isinstance(story, str):
                raise TypeError(f"stories[{i}] must be a string.")
            normalized = self._normalize(story)
            normalized_stories.append(normalized)
            names_info_per_story.append(self._extract_person_names(normalized))

        # Global name uniqueness across the provided set (only if extraction is available for all stories).
        all_have_names_available = all(bool(info["available"]) for info in names_info_per_story)
        repeated_names_global: Optional[set] = None
        if all_have_names_available:
            freq: Dict[str, int] = {}
            for info in names_info_per_story:
                for n in set(info["names"]):
                    freq[n] = freq.get(n, 0) + 1
            repeated_names_global = {name for name, count in freq.items() if count > 1}

        story_reports: List[Dict[str, object]] = []
        total_violations = 0
        total_checked = 0
        total_not_checked = 0

        for idx, (normalized, names_info) in enumerate(zip(normalized_stories, names_info_per_story)):
            results = self._evaluate_single_story(
                text=normalized,
                extracted_names=names_info["names"],
                names_available=bool(names_info["available"]),
                name_unavailable_reason=names_info.get("reason"),
                repeated_names_global=repeated_names_global,
            )

            report = self._build_story_output(normalized, names_info, results)
            report["story_index"] = idx
            story_reports.append(report)

            total_violations += int(report["violations_count"])
            total_checked += int(report["checked_constraints_count"])
            total_not_checked += int(report["not_checked_constraints_count"])

        return {
            "num_stories": len(story_reports),
            "constraints_per_story": len(self.CONSTRAINTS),
            "total_constraints_evaluated": total_checked,
            "total_constraints_not_checked": total_not_checked,
            "total_violations": total_violations,
            "mean_violations_per_story": total_violations / len(story_reports),
            "repeated_names_across_stories": sorted(list(repeated_names_global or set())),
            "story_reports": story_reports,
        }

    def count_violations(self, stories: Sequence[str]) -> int:
        return int(self.evaluate_stories(stories)["total_violations"])

    def print_report(self, stories: Sequence[str]) -> Dict[str, object]:
        result = self.evaluate_stories(stories)

        print("=== EGRA Constraint Adherence Report (High-Confidence Mode) ===")
        print(f"Stories: {result['num_stories']}")
        print(f"Constraints/story: {result['constraints_per_story']}")
        print(f"Checked constraints total: {result['total_constraints_evaluated']}")
        print(f"Not checked constraints total: {result['total_constraints_not_checked']}")
        print(f"Total violations (checked only): {result['total_violations']}")
        print(f"Mean violations/story: {result['mean_violations_per_story']:.2f}")
        print(f"Repeated names across stories: {result['repeated_names_across_stories']}")

        for report in result["story_reports"]:
            print(
                f"- Story #{report['story_index']}: violations={report['violations_count']}, "
                f"checked={report['checked_constraints_count']}, "
                f"not_checked={report['not_checked_constraints_count']}, "
                f"violated={report['violated_constraint_ids']}"
            )

        return result
