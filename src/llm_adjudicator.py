"""
Step 8: send the undecided pairs -- and only those -- to an LLM.

The point of this stage is not that an LLM can compare two records. Of course
it can. The point is WHERE you spend it.

The classifier already decides ~96% of pairs at effectively zero marginal
cost. Those are the easy ones: identical phone numbers, or nothing whatsoever
in common. Paying a language model to look at them buys nothing. The
remaining few percent are genuinely ambiguous -- a nickname plus a house move
plus a missing date of birth -- and that is where a model that understands
that "Raju" is short for "Rajesh", and that Secunderabad is part of Hyderabad,
earns its keep.

So this script measures three things and reports them together:
    1. Does the LLM actually agree with ground truth on the hard cases?
    2. What does it cost per thousand pairs?
    3. What would it have cost to send everything?

That comparison is the whole argument, and it is arithmetic, not magic.

Requires ANTHROPIC_API_KEY in the environment.

Run:  python -m src.llm_adjudicator --dry-run     estimate cost, call nothing
      python -m src.llm_adjudicator               adjudicate for real
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import config
from src.load import load_standardized

CACHE_PATH = config.DATA_PROCESSED / "llm_verdicts.csv"

PROMPT_TEMPLATE = """You are helping a bank deduplicate its customer database.

Decide whether these two records describe the SAME real person.

Record A ({system_a}):
  Name:    {name_a}
  DOB:     {dob_a}
  Email:   {email_a}
  Phone:   {phone_a}
  Address: {address_a}, {city_a} {pincode_a}

Record B ({system_b}):
  Name:    {name_b}
  DOB:     {dob_b}
  Email:   {email_b}
  Phone:   {phone_b}
  Address: {address_b}, {city_b} {pincode_b}

Context that matters for Indian customer data:
- Short forms are common: Raju/Rajesh, Sanju/Sanjay, Abhi/Abhishek.
- Name order varies. "Sharma Rajesh" and "Rajesh Sharma" are one person.
- Initials replace full names: "R. K. Sharma" may be "Rajesh Kumar Sharma".
- Cities have several names: Bengaluru/Bangalore, Hyderabad/Secunderabad.
- People move house, so a different address does NOT rule out a match.
- Families share phone numbers and addresses, so a shared phone does NOT
  prove a match. Two different first names at one address are usually two
  different people.
- Dates may be written day-first or month-first, so 03/11 and 11/03 can be
  the same date.

Reply with ONLY a JSON object, no other text:
{{"match": true or false, "confidence": 0.0 to 1.0, "reason": "one short sentence"}}"""


def format_pair(row, records: pd.DataFrame) -> str:
    a = records.loc[row.pos_a]
    b = records.loc[row.pos_b]

    def name(record):
        parts = [record.first_name, record.middle_name, record.last_name]
        return " ".join(p for p in parts if p) or "(missing)"

    return PROMPT_TEMPLATE.format(
        system_a=a.source_system, system_b=b.source_system,
        name_a=name(a), name_b=name(b),
        dob_a=a.dob or "(missing)", dob_b=b.dob or "(missing)",
        email_a=a.email or "(missing)", email_b=b.email or "(missing)",
        phone_a=a.phone or "(missing)", phone_b=b.phone or "(missing)",
        address_a=a.address_line or "(missing)", address_b=b.address_line or "(missing)",
        city_a=a.city, city_b=b.city,
        pincode_a=a.pincode, pincode_b=b.pincode,
    )


def parse_verdict(text: str) -> dict:
    """Pull the JSON object out of the reply, tolerating stray prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"match": None, "confidence": None, "reason": "unparseable reply"}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"match": None, "confidence": None, "reason": "invalid json"}
    return {
        "match": payload.get("match"),
        "confidence": payload.get("confidence"),
        "reason": str(payload.get("reason", ""))[:200],
    }


def adjudicate(client, prompt: str) -> tuple[dict, int, int]:
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = parse_verdict(response.content[0].text)
    return verdict, response.usage.input_tokens, response.usage.output_tokens


def cost_of(input_tokens: int, output_tokens: int) -> float:
    """Cost in INR for the given token counts, at the configured rates."""
    usd = (input_tokens / 1_000_000 * config.LLM_INPUT_COST_PER_MTOK
           + output_tokens / 1_000_000 * config.LLM_OUTPUT_COST_PER_MTOK)
    return usd * config.USD_TO_INR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate cost and print one prompt, call nothing")
    parser.add_argument("--limit", type=int, default=config.LLM_MAX_PAIRS)
    args = parser.parse_args()

    decisions = pd.read_csv(config.DATA_PROCESSED / "decisions.csv")
    records = load_standardized()
    review = decisions[decisions.band == "review"].copy()

    total_pairs = len(decisions)
    print(f"{total_pairs:,} candidate pairs total")
    print(f"{len(review):,} in the review band ({len(review) / total_pairs:.2%}) "
          f"-- these are the only ones we send\n")

    sample = review.sample(n=min(args.limit, len(review)),
                           random_state=config.RANDOM_SEED)
    prompts = [format_pair(row, records) for row in sample.itertuples()]

    if args.dry_run:
        print("=" * 74)
        print("EXAMPLE PROMPT")
        print("=" * 74)
        print(prompts[0])
        # ~4 characters per token is the usual rough rule for English text.
        est_in = sum(len(p) for p in prompts) // 4 // len(prompts)
        print("\n" + "=" * 74)
        print(f"Estimated ~{est_in:,} input tokens per pair, ~60 output")
        print(f"Sending {len(sample):,} review pairs would cost about "
              f"Rs {cost_of(est_in * len(sample), 60 * len(sample)):.2f}")
        print(f"Sending all {total_pairs:,} pairs would cost about "
              f"Rs {cost_of(est_in * total_pairs, 60 * total_pairs):,.2f}")
        print("=" * 74)
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set.\n"
            "  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "Or run with --dry-run to see the prompt and a cost estimate "
            "without calling anything."
        )

    from anthropic import Anthropic
    client = Anthropic()

    print(f"Adjudicating {len(sample):,} pairs with {config.LLM_MODEL}...")
    results, in_tokens, out_tokens = [], 0, 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (verdict, used_in, used_out) in enumerate(
            pool.map(lambda p: adjudicate(client, p), prompts), start=1
        ):
            results.append(verdict)
            in_tokens += used_in
            out_tokens += used_out
            if i % 25 == 0:
                print(f"  {i:,}/{len(prompts):,}")

    sample["llm_match"] = [r["match"] for r in results]
    sample["llm_confidence"] = [r["confidence"] for r in results]
    sample["llm_reason"] = [r["reason"] for r in results]
    sample.to_csv(CACHE_PATH, index=False)

    usable = sample[sample.llm_match.notna()].copy()
    usable["llm_says_match"] = usable.llm_match.astype(bool).astype(int)
    agreement = (usable.llm_says_match == usable.is_match).mean()

    true_positive = int(((usable.llm_says_match == 1) & (usable.is_match == 1)).sum())
    false_positive = int(((usable.llm_says_match == 1) & (usable.is_match == 0)).sum())
    false_negative = int(((usable.llm_says_match == 0) & (usable.is_match == 1)).sum())

    # What the classifier alone would have done with these same pairs, had it
    # been forced to choose instead of escalating them.
    forced = (sample.score >= 0.5).astype(int)
    forced_agreement = (forced == sample.is_match).mean()

    spent = cost_of(in_tokens, out_tokens)
    per_pair = spent / len(sample)

    print("\n" + "=" * 74)
    print("LLM ADJUDICATION ON THE REVIEW BAND")
    print("=" * 74)
    print(f"  Pairs adjudicated            {len(usable):>10,}")
    print(f"  Agreement with ground truth  {agreement:>10.1%}")
    print(f"  Classifier forced to guess   {forced_agreement:>10.1%}  "
          f"(same pairs, threshold 0.5)")
    print(f"  Improvement                  {agreement - forced_agreement:>+10.1%}")
    print(f"\n  Correctly matched            {true_positive:>10,}")
    print(f"  Wrongly matched              {false_positive:>10,}")
    print(f"  Wrongly separated            {false_negative:>10,}")

    print("\n" + "=" * 74)
    print("COST: SELECTIVE vs EVERYTHING")
    print("=" * 74)
    print(f"  Tokens used                  {in_tokens:>10,} in / {out_tokens:,} out")
    print(f"  Cost for {len(sample):,} pairs         Rs {spent:>10.2f}")
    print(f"  Cost per pair                Rs {per_pair:>10.4f}")
    print()
    review_cost = per_pair * len(review)
    all_cost = per_pair * total_pairs
    print(f"  Review band only ({len(review):,} pairs)   Rs {review_cost:>10,.2f}")
    print(f"  Every pair ({total_pairs:,} pairs)     Rs {all_cost:>10,.2f}")
    print(f"  Ratio                        {all_cost / review_cost:>10.1f}x more expensive")
    print()
    print("  The classifier already decides the other "
          f"{1 - len(review) / total_pairs:.1%} correctly and for free.")
    print("  Paying a language model to re-confirm those is the entire waste.")
    print(f"\nWrote verdicts -> {CACHE_PATH.name}")


if __name__ == "__main__":
    main()
