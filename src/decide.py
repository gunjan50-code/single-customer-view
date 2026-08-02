"""
Step 6 of the pipeline: turn scores into decisions.

A single threshold at 0.5 would be the obvious thing to do and the wrong one.
It forces the system to guess on the pairs it is least sure about, which are
exactly the pairs where a mistake is most expensive.

Instead we use two thresholds and three outcomes -- the classical
Fellegi-Sunter design from 1969, which is still what commercial master data
management tools implement:

    score >= UPPER   ->  auto-merge      no human involved
    score <= LOWER   ->  auto-reject     no human involved
    in between       ->  clerical review a person decides

The business metric this produces is the STRAIGHT-THROUGH PROCESSING RATE:
the share of decisions made without a human. That is the number a client is
actually buying, because it translates directly into headcount.

The two thresholds are not symmetric, and should not be. A missed duplicate
is an annoyance that a later run can catch. A wrong merge fuses two real
customers -- their balances, their KYC status, their credit history -- and is
an incident that is expensive and embarrassing to unpick. So the auto-merge
threshold is tuned for precision, and recall is allowed to suffer.

Run:  python -m src.decide
"""

import joblib
import numpy as np
import pandas as pd

import config
from src.features import FEATURE_NAMES

AUTO_MERGE = "auto_merge"
AUTO_REJECT = "auto_reject"
REVIEW = "review"


def assign_bands(scores: np.ndarray, upper: float, lower: float) -> np.ndarray:
    bands = np.full(len(scores), REVIEW, dtype=object)
    bands[scores >= upper] = AUTO_MERGE
    bands[scores <= lower] = AUTO_REJECT
    return bands


def band_report(frame: pd.DataFrame, label: str) -> None:
    """Break results down by band, and count the mistakes that matter."""
    total = len(frame)
    print(f"\n  {label}  ({total:,} pairs)")
    print(f"    {'band':<14} {'pairs':>9} {'share':>8} {'true matches':>14} "
          f"{'purity':>9}")

    for band in (AUTO_MERGE, REVIEW, AUTO_REJECT):
        subset = frame[frame.band == band]
        if subset.empty:
            print(f"    {band:<14} {0:>9,}")
            continue
        n_true = int(subset.is_match.sum())
        purity = n_true / len(subset)
        print(f"    {band:<14} {len(subset):>9,} {len(subset) / total:>7.1%} "
              f"{n_true:>14,} {purity:>8.1%}")

    merged = frame[frame.band == AUTO_MERGE]
    rejected = frame[frame.band == AUTO_REJECT]
    reviewed = frame[frame.band == REVIEW]

    wrong_merges = int((merged.is_match == 0).sum())
    missed = int((rejected.is_match == 1).sum())
    automated = len(merged) + len(rejected)

    print(f"\n    Straight-through rate   {automated / total:>7.1%}  "
          f"({automated:,} of {total:,} decided with no human)")
    print(f"    Sent to human review    {len(reviewed) / total:>7.1%}  "
          f"({len(reviewed):,} pairs)")
    print(f"    WRONG MERGES            {wrong_merges:>7,}  "
          f"(auto-merged but different people)")
    print(f"    Missed duplicates       {missed:>7,}  "
          f"(auto-rejected but the same person)")
    if len(merged):
        print(f"    Auto-merge precision    {1 - wrong_merges / len(merged):>7.4%}")


def sweep_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    """What do different threshold choices actually buy you?

    This is the table to put in front of a client. It makes the trade explicit
    -- more automation always costs accuracy, and the business gets to choose
    where on that curve it wants to sit, rather than inheriting whatever
    0.5 happened to give.
    """
    rows = []
    for upper in (0.80, 0.90, 0.95, 0.99, 0.995):
        for lower in (0.05, 0.10, 0.30):
            bands = assign_bands(frame.score.to_numpy(), upper, lower)
            merged = frame[bands == AUTO_MERGE]
            rejected = frame[bands == AUTO_REJECT]
            automated = len(merged) + len(rejected)
            wrong = int((merged.is_match == 0).sum())
            missed = int((rejected.is_match == 1).sum())
            rows.append({
                "upper": upper,
                "lower": lower,
                "straight_through": automated / len(frame),
                "review_share": 1 - automated / len(frame),
                "wrong_merges": wrong,
                "missed_duplicates": missed,
                "merge_precision": (1 - wrong / len(merged)) if len(merged) else float("nan"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    model = joblib.load(config.DATA_PROCESSED / "matcher.joblib")
    pairs = pd.read_csv(config.DATA_PROCESSED / "pair_features.csv")
    test_people = set(pd.read_csv(config.DATA_PROCESSED / "test_people.csv").person_id)

    # In production the model scores everything, so that is what we do here.
    # But we report accuracy separately on the held-out people, because
    # scoring the pairs the model trained on would flatter it.
    print(f"Scoring all {len(pairs):,} candidate pairs with the saved model...")
    pairs["score"] = model.predict_proba(pairs[FEATURE_NAMES])[:, 1]
    pairs["band"] = assign_bands(
        pairs.score.to_numpy(), config.UPPER_THRESHOLD, config.LOWER_THRESHOLD
    )

    is_test = pairs.person_a.isin(test_people) & pairs.person_b.isin(test_people)

    print("\n" + "=" * 74)
    print("DECISION BANDS")
    print(f"  auto-merge at >= {config.UPPER_THRESHOLD}, "
          f"auto-reject at <= {config.LOWER_THRESHOLD}")
    print("=" * 74)
    band_report(pairs[is_test], "HELD-OUT PEOPLE (the honest number)")
    band_report(pairs, "ALL PAIRS (what actually gets written to the database)")

    print("\n" + "=" * 74)
    print("CHOOSING THE THRESHOLDS: WHAT EACH CHOICE COSTS")
    print("  measured on held-out people only")
    print("=" * 74)
    sweep = sweep_thresholds(pairs[is_test])
    print(f"  {'upper':>6} {'lower':>6} {'straight-thru':>14} {'review':>8} "
          f"{'wrong merges':>13} {'missed dups':>12} {'merge prec':>11}")
    for row in sweep.itertuples():
        print(f"  {row.upper:>6.3f} {row.lower:>6.2f} {row.straight_through:>13.1%} "
              f"{row.review_share:>8.1%} {row.wrong_merges:>13,} "
              f"{row.missed_duplicates:>12,} {row.merge_precision:>10.3%}")
    sweep.to_csv(config.REPORTS / "threshold_sweep.csv", index=False)

    columns = ["record_id_a", "record_id_b", "pos_a", "pos_b",
               "person_a", "person_b", "is_match", "score", "band"]
    pairs[columns].to_csv(config.DATA_PROCESSED / "decisions.csv", index=False)
    print(f"\nWrote {len(pairs):,} decisions -> decisions.csv")
    print(f"  {int((pairs.band == REVIEW).sum()):,} pairs queued for human review")


if __name__ == "__main__":
    main()
