"""Block 2. One call, and everything you can learn from it.

    python 01_first_call.py

A call that prints only the answer is not finished. By the end of this file
you print four things and can explain all four, and you write the run into
`artifacts/traces.jsonl` in the shape every later week reads.

Three TODO markers. Do them in order.
"""

from __future__ import annotations

import time

from openai import OpenAI

from project.models import BASE_URL, API_KEY, SMALL
from project.prices import estimate
from project.trace import TraceRecorder, local_conditions

QUESTION = ("A resident asks how to register a change of address. "
            "Answer in two sentences.")


def main() -> int:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    rec = TraceRecorder(
        week=1,
        case_id="W1-first-call",
        conditions=local_conditions(SMALL.name, temperature=0.0),
        user_input=QUESTION,
    )

    with rec.step("model", SMALL.name) as step:
        started = time.perf_counter()

        # TODO 1. Make the call.
        reply = client.chat.completions.create(
            model=SMALL.name,
            messages=[{"role": "user", "content": QUESTION}],
            temperature=0.0,
            max_tokens=200,
        )

        elapsed = time.perf_counter() - started

        if reply is None:
            print("TODO 1 is not done yet: `reply` is still None.")
            print("Open this file and make the call. The four lines you "
                  "need are in the comment above.")
            return 1

        step.tokens(reply.usage.prompt_tokens, reply.usage.completion_tokens)
        step.detail(finish_reason=reply.choices[0].finish_reason)

    # TODO 2. Print four things, and be ready to say what each one means.
    answer = reply.choices[0].message.content
    print("\n--- TODO 2: print the four things here ---\n")
    print(f"a. Answer text:\n{answer}\n")
    print(f"b. Finish reason: {reply.choices[0].finish_reason}")
    print(f"c. Token counts - Prompt: {reply.usage.prompt_tokens}, Completion: {reply.usage.completion_tokens}")
    print(f"d. Elapsed time: {elapsed:.4f} seconds")

    # TODO 3. Close the trace.
    rec.finish(
        output=answer,
        outcome="ok",
    )

    if reply is not None:
        est = estimate(reply.usage.prompt_tokens,
                       reply.usage.completion_tokens, tier="small")
        print(est.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
