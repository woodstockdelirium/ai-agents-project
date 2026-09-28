"""Block 2. The zero-shot baseline, scored per field.

    python 01_zero_shot.py --replay     # the shipped recording, instant
    python 01_zero_shot.py              # your own model, about 45 seconds

Develop your scorer against `--replay`. The recording holds every model
answer for both variants, so your scorer runs in well under a second and you
can iterate on it properly instead of waiting forty-five seconds to find out
you compared the wrong field.

The recording contains real failures, because the model really does make
them. If your scorer reports forty out of forty, your scorer does nothing.

One TODO marker here. TODO 1 to 4 live in extractor.py and scoring.py, and
this file will not run until they are done.
"""

from __future__ import annotations

import argparse

from documents import DOCS, GOLD
from extractor import (PROMPT_VERSION, SYSTEM_ZERO_SHOT, get_client,
                       run_variant)

from project.trace import write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()

    client = get_client(args.replay)
    board, records, metas = run_variant(client, SYSTEM_ZERO_SHOT,
                                        "zero-shot", DOCS, GOLD)

    if board.failures:
        print("failures worth reading:")
        for doc_id, fieldname, note in board.failures[:10]:
            print(f"  {doc_id}  {fieldname:<9} {note}")

    write_json("artifacts/week02_zero_shot.json", {
        "variant": "zero-shot",
        "prompt_version": PROMPT_VERSION,
        "hits": board.hits, "total": board.total, "invalid": board.invalid,
    })

    # TODO 7. Write the gold set into the project spine.
    cases = []
    for doc, gold in zip(DOCS, GOLD):
        # Safe extraction whether gold is a dict, string, or object
        if isinstance(gold, str):
            import json
            try:
                gold_dict = json.loads(gold)
            except Exception:
                gold_dict = {}
        elif isinstance(gold, dict):
            gold_dict = gold
        else:
            gold_dict = gold.model_dump() if hasattr(gold, "model_dump") else gold.__dict__

        due_date = gold_dict.get("due_date") if isinstance(gold_dict, dict) else getattr(gold, "due_date", None)
        category = gold_dict.get("category") if isinstance(gold_dict, dict) else getattr(gold, "category", None)
        urgency = gold_dict.get("urgency") if isinstance(gold_dict, dict) else getattr(gold, "urgency", None)

        date_desc = f"due date {due_date}" if due_date else "no explicit calendar date"
        expected_behavior = (
            f"Extracts category '{category}' and urgency '{urgency}', "
            f"with {date_desc}."
        )
        cases.append({
            "case_id": doc.id,
            "week_added": 2,
            "question": doc.text,
            "expected": gold_dict if isinstance(gold_dict, dict) else gold.model_dump(),
            "expected_behavior": expected_behavior,
            "slice_tags": [doc.lang],
        })

    write_json("artifacts/goldset.json", {"cases": cases})


if __name__ == "__main__":
    raise SystemExit(main())
