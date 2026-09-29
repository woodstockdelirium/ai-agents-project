from __future__ import annotations

from openai import OpenAI
from queries import QUERIES
from router import classify
from project.models import BASE_URL, API_KEY, SMALL

def main() -> int:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    print(f"{'ID':<6} {'True Route':<12} {'Predicted':<12} {'Confidence'}")
    print("-" * 45)
    
    for q in QUERIES:
        decision, meta = classify(client, q.text, model=SMALL.name)
        if decision:
            print(f"{q.id:<6} {q.route:<12} {decision.route:<12} {decision.confidence:<12.2f}")
        else:
            print(f"{q.id:<6} {q.route:<12} {'FAILED':<12} {'N/A'}")
            
    return 0

if __name__ == "__main__":
    raise SystemExit(main())