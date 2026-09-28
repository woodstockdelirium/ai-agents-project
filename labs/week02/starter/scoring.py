"""The scorer. Two TODO markers, and it is the most important file today.

A prompt change is not an improvement until it has been measured, and this
is what measures it. Write it before you tune anything, because a scorer
written after you have seen the output tends to score what the output
already does.

One rule, and it decides most of the marks in this session: report per
field, as counts. Never one overall accuracy. Ten records means one error
moves a percentage by ten points, and an average across four fields hides
the only interesting thing in the data, which is that they do not move
together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FIELDS = ("category", "urgency", "due_date", "quote")


@dataclass
class FieldResult:
    correct: bool
    got: object
    expected: object
    note: str = ""


@dataclass
class Scoreboard:
    """Counts per field, plus the failures worth reading."""

    hits: dict[str, int] = field(
        default_factory=lambda: {f: 0 for f in FIELDS})
    total: int = 0
    invalid: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)

    def as_counts(self) -> str:
        return "  ".join(f"{f} {self.hits[f]:>2}/{self.total}"
                         for f in FIELDS)


# --------------------------------------------------------------------------
# TODO 3. Score one record against its gold annotation.
# --------------------------------------------------------------------------

def score_one(record, gold, document_text: str) -> dict[str, FieldResult]:
    results = {}
    
    # Helper to safely extract field values whether gold is an object, dict, or json string
    def get_val(obj, field):
        if isinstance(obj, str):
            import json
            try:
                obj = json.loads(obj)
            except Exception:
                pass
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    # If the model failed parsing entirely, all fields fail
    if record is None:
        for f in FIELDS:
            results[f] = FieldResult(correct=False, got=None, expected=get_val(gold, f), note="validation failed")
        return results

    # 1. Category
    gold_cat = get_val(gold, "category")
    cat_correct = record.category == gold_cat
    results["category"] = FieldResult(correct=cat_correct, got=record.category, expected=gold_cat)

    # 2. Urgency
    gold_urg = get_val(gold, "urgency")
    urg_correct = record.urgency == gold_urg
    results["urgency"] = FieldResult(correct=urg_correct, got=record.urgency, expected=gold_urg)

    # 3. Due Date
    gold_date = get_val(gold, "due_date")
    got_date = record.due_date if record.due_date not in ("", "null", "None") else None
    exp_date = gold_date if gold_date not in ("", "null", "None") else None
    date_correct = got_date == exp_date
    results["due_date"] = FieldResult(correct=date_correct, got=record.due_date, expected=gold_date)

    # 4. Quote
    gold_quote = get_val(gold, "quote")
    quote_correct = bool(record.quote and record.quote in document_text)
    note = "" if quote_correct else "quote not found verbatim in document"
    results["quote"] = FieldResult(correct=quote_correct, got=record.quote, expected=gold_quote, note=note)

    return results


# --------------------------------------------------------------------------
# TODO 4. Aggregate.
# --------------------------------------------------------------------------

def score_all(records, golds, docs) -> Scoreboard:
    scoreboard = Scoreboard()
    scoreboard.total = len(docs)

    for record, gold, doc in zip(records, golds, docs):
        if record is None:
            scoreboard.invalid += 1
            # Every field counts as wrong for this invalid record
            for f in FIELDS:
                scoreboard.failures.append((doc.id, f, "record parsing failed validation"))
            continue

        res = score_one(record, gold, doc.text)
        for f in FIELDS:
            if res[f].correct:
                scoreboard.hits[f] += 1
            else:
                scoreboard.failures.append((doc.id, f, res[f].note or f"got {res[f].got!r}, expected {res[f].expected!r}"))

    return scoreboard


# --------------------------------------------------------------------------
# Given.
# --------------------------------------------------------------------------

def compare(a: Scoreboard, b: Scoreboard, label_a: str, label_b: str) -> str:
    """Two scoreboards side by side, per field, with the movement."""
    lines = [f"{'field':<10} {label_a:>12} {label_b:>12} {'move':>7}"]
    lines.append("-" * 44)
    for f in FIELDS:
        move = b.hits[f] - a.hits[f]
        lines.append(f"{f:<10} {a.hits[f]:>9}/{a.total} {b.hits[f]:>9}/{b.total} "
                     f"{move:>+7d}")
    lines.append(f"{'invalid':<10} {a.invalid:>12} {b.invalid:>12}")
    return "\n".join(lines)
