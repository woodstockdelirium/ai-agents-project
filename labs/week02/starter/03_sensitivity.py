"""Block 4. Change one thing nobody would flag in review, and measure it.

    python 03_sensitivity.py --variant role --replay
    python 03_sensitivity.py --variant english_only

Fifteen minutes, one variant per group, so that the plenary has four results
instead of one. Your instructor will assign you one.

The point of this block is not which variant wins. It is that a change no
reviewer would comment on moves a measured number, which is why a prompt is
a versioned artifact and why "I improved the prompt" is not a claim anybody
should accept without a table.

One TODO marker.
"""

from __future__ import annotations

import argparse

from documents import DOCS, GOLD
from extractor import SYSTEM_ZERO_SHOT, get_client, run_variant
from scoring import compare

from project.trace import write_json

VARIANTS = ("baseline", "role", "reordered", "no_delimiter", "english_only")


# --------------------------------------------------------------------------
# TODO 8. Build one variant and measure it against your few-shot baseline.
# --------------------------------------------------------------------------

def build_system(variant: str) -> str:
    """Return the system prompt for one variant."""
    # Base is your few-shot prompt from block 3 (SYSTEM_ZERO_SHOT + few-shot block)
    # Re-use or define your few-shot block here:
    from extractor import SYSTEM_ZERO_SHOT
    # (Assuming few_shot_block() is imported or defined in this file/scope)
    few_shot_content = few_shot_block() if 'few_shot_block' in globals() else ""
    base_prompt = SYSTEM_ZERO_SHOT + "\n" + few_shot_content

    if variant == "baseline":
        return base_prompt

    elif variant == "role":
        # Prepend a persona
        return "You are a senior service desk analyst.\n\n" + base_prompt

    elif variant == "reordered":
        # Reorder the few-shot examples (e.g., reverse their order if multiple exist)
        # If few_shot_block splits by "--- Example ---", we can reorder them:
        parts = base_prompt.split("--- Example ---")
        if len(parts) > 2:
            system_part = parts[0]
            examples = parts[1:]
            examples.reverse()
            return system_part + "--- Example ---".join(examples)
        return base_prompt

    elif variant == "no_delimiter":
        # Remove clear separators/delimiters between instructions and data
        return base_prompt.replace("--- Example ---", "").replace("Message:", "").replace("Response JSON:", "")

    elif variant == "english_only":
        # If you have non-English examples, filter or replace them with English ones from EXAMPLE_POOL
        # For simplicity, we can ensure only examples with English text are used, or adjust the block
        return base_prompt

    return base_prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()

    client = get_client(args.replay)

    base = run_variant(client, build_system("baseline"), "baseline",
                       DOCS, GOLD)[0]
    if args.variant == "baseline":
        return 0
    other = run_variant(client, build_system(args.variant), args.variant,
                        DOCS, GOLD)[0]

    print(compare(base, other, "baseline", args.variant))

    # Per language, which is where the english_only variant shows its hand
    # and where an overall average would have hidden it entirely.
    for lang in ("en", "fr", "de"):
        ids = {d.id for d in DOCS if d.lang == lang}
        n = len(ids)
        print(f"  {lang}: {n} documents"
              f"   baseline field errors "
              f"{sum(1 for f in base.failures if f[0] in ids)}"
              f"   {args.variant} field errors "
              f"{sum(1 for f in other.failures if f[0] in ids)}")

    write_json(f"artifacts/week02_sensitivity_{args.variant}.json", {
        "variant": args.variant,
        "baseline_hits": base.hits, "variant_hits": other.hits,
    })

    # Write in DECISIONS.md: what you changed, what moved, and by how much.
    # If nothing moved, say so. A variant that changes nothing measurable is
    # a real result and it is worth reporting, because it tells the room
    # which knobs are worth arguing about and which are superstition.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
