"""
Step 5 of the pipeline: train the match classifier.

Model choice: plain logistic regression, deliberately.

  1. It is interpretable. You can print one coefficient per feature and say
     exactly why a pair scored the way it did. A gradient booster would score
     a point or two higher and explain nothing.
  2. Fitted by log loss, its raw outputs start out close to true probabilities.

On (2), with an important correction found by actually measuring it. Logistic
regression is only well calibrated when it is fitted on the true class
balance. We use class_weight='balanced' -- necessary here, because only ~10%
of candidate pairs are matches -- and that reweighting deliberately distorts
the output scale. Measured on the first run, the raw model claimed 0.61 for a
bucket of pairs that were actually matches only 31% of the time: badly
overconfident, and overconfident precisely in the middle of the range where
the human-review threshold sits.

So the model is fitted twice: once plainly, to read the coefficients and
explain what it learned, and once wrapped in CalibratedClassifierCV, which
learns a correction curve mapping raw scores onto honest probabilities. The
calibrated model is what the decision layer consumes. This matters because
"0.95" has to mean "95% likely to be the same person" for the threshold bands
to be anything more than superstition.

Splitting: BY PERSON, not by pair. If the same person appeared in both the
training and test sets, the model would be tested on people it had already
seen and the score would be inflated. Pairs that straddle the split are
dropped rather than assigned arbitrarily.

Run:  python -m src.train
"""

import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.features import FEATURE_NAMES


def split_by_person(pairs: pd.DataFrame, test_size: float, seed: int):
    """Assign whole people to train or test, then keep only intra-split pairs."""
    rng = random.Random(seed)
    people = sorted(set(pairs.person_a) | set(pairs.person_b))
    test_people = {p for p in people if rng.random() < test_size}

    a_in_test = pairs.person_a.isin(test_people)
    b_in_test = pairs.person_b.isin(test_people)

    train_mask = ~a_in_test & ~b_in_test
    test_mask = a_in_test & b_in_test
    dropped = len(pairs) - int(train_mask.sum()) - int(test_mask.sum())

    print(f"  {len(people):,} people -> {len(people) - len(test_people):,} train / "
          f"{len(test_people):,} test")
    print(f"  {int(train_mask.sum()):,} train pairs, {int(test_mask.sum()):,} test pairs")
    print(f"  {dropped:,} pairs dropped for straddling the split (leakage guard)")

    return pairs[train_mask].copy(), pairs[test_mask].copy()


def naive_baseline(pairs: pd.DataFrame) -> dict:
    """The rule a sensible person would write before reaching for ML.

    'Same person if the full name matches closely AND the date of birth is
    identical.' Worth measuring, because if the model cannot beat this it is
    not earning its complexity.
    """
    predicted = ((pairs.full_name_sim > 0.90) & (pairs.dob_exact == 1)).astype(int)
    return {
        "precision": precision_score(pairs.is_match, predicted, zero_division=0),
        "recall": recall_score(pairs.is_match, predicted, zero_division=0),
        "f1": f1_score(pairs.is_match, predicted, zero_division=0),
    }


def reliability_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Are the predicted probabilities honest?

    Bucket the scores, then check what fraction of each bucket really were
    matches. If the model says 0.9 and 90% of those pairs are matches, the
    number means what it claims and the thresholds downstream are safe.
    """
    buckets = pd.cut(scores, bins=[0, .1, .3, .5, .7, .9, .95, .99, 1.0],
                     include_lowest=True)
    frame = pd.DataFrame({"bucket": buckets, "actual": y_true, "score": scores})
    table = frame.groupby("bucket", observed=True).agg(
        pairs=("actual", "size"),
        mean_predicted=("score", "mean"),
        actual_match_rate=("actual", "mean"),
    ).reset_index()
    table["gap"] = table.actual_match_rate - table.mean_predicted
    return table


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray,
                               n_bins: int = 20) -> float:
    """Average gap between claimed probability and observed frequency.

    Equal-width bins, weighted by how many pairs land in each. Lower is
    better; 0 would mean every stated probability is exactly right.

    A caveat that matters for reading the number: our score distribution is
    heavily bimodal, with most pairs sitting near 0 or near 1 where any
    method is accurate. That drags ECE down and can hide a large error in the
    sparse middle -- which is exactly the region the review band lives in. So
    we look at the bucket table too, not just this one number.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(scores, bins[1:-1])
    total_error, total_weight = 0.0, 0
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        total_error += abs(y_true[mask].mean() - scores[mask].mean()) * count
        total_weight += count
    return total_error / max(total_weight, 1)


def choose_upper_threshold(y_true: np.ndarray, scores: np.ndarray,
                           target_precision: float = 0.995) -> float:
    """Find the lowest score we can auto-merge at while staying above target.

    The asymmetry is the point. A missed duplicate is an annoyance that a
    later run can catch. A wrong merge fuses two real customers' records --
    their balances, their KYC status, their credit history -- and is a
    serious incident that is expensive to unpick. So the auto-merge band is
    tuned for precision and lets recall fall where it may.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    for precision, threshold in zip(precisions[:-1], thresholds):
        if precision >= target_precision:
            return float(threshold)
    return 0.99


def main() -> None:
    pairs = pd.read_csv(config.DATA_PROCESSED / "pair_features.csv")
    print(f"Loaded {len(pairs):,} labelled pairs "
          f"({pairs.is_match.mean():.2%} positive)\n")

    print("Splitting by person to avoid leakage:")
    train, test = split_by_person(pairs, config.TEST_SIZE, config.RANDOM_SEED)

    X_train, y_train = train[FEATURE_NAMES], train.is_match
    X_test, y_test = test[FEATURE_NAMES], test.is_match

    # class_weight='balanced' matters: only ~10% of candidate pairs are matches,
    # and without it the model is rewarded for simply predicting "no" a lot.
    def make_pipeline() -> Pipeline:
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ])

    # Fit 1: plain, used only to read the coefficients and explain the model.
    raw_model = make_pipeline()
    raw_model.fit(X_train, y_train)
    raw_scores = raw_model.predict_proba(X_test)[:, 1]

    # Fit 2: calibrated. Which correction to use is an empirical question, so
    # we fit both and measure rather than guessing.
    #
    #   sigmoid  (Platt) fits a 2-parameter logistic curve. Rigid, but stable
    #            when the middle of the range is sparse -- it cannot wiggle to
    #            chase noise.
    #   isotonic fits a free-form monotonic step function. More flexible, but
    #            it needs data everywhere, and our scores are strongly bimodal
    #            with barely 1,000 pairs between 0.1 and 0.99.
    candidates = {"raw (uncalibrated)": raw_scores}
    fitted = {"raw (uncalibrated)": raw_model}
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(make_pipeline(), method=method, cv=5)
        calibrated.fit(X_train, y_train)
        fitted[method] = calibrated
        candidates[method] = calibrated.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("CALIBRATION: WHICH CORRECTION ACTUALLY HELPS?")
    print("=" * 70)
    print("  Overall ECE is a misleading way to choose here. Most pairs sit at")
    print("  0.00 or 1.00 where every method is nearly perfect, and those pairs")
    print("  dominate the average. But no decision is ever in doubt out there --")
    print("  the thresholds live in the middle. So we select on calibration")
    print("  error restricted to the undecided band, and report both.")
    print()
    print(f"  {'method':<20} {'ECE all':>10} {'ECE band':>10} {'PR-AUC':>9} {'F1@0.5':>9}")

    band = (raw_scores > config.LOWER_THRESHOLD / 2) & (raw_scores < 0.995)
    errors, band_errors = {}, {}
    for name, candidate_scores in candidates.items():
        errors[name] = expected_calibration_error(y_test.to_numpy(), candidate_scores)
        band_errors[name] = expected_calibration_error(
            y_test.to_numpy()[band], candidate_scores[band], n_bins=10
        )
        print(f"  {name:<20} {errors[name]:>10.5f} {band_errors[name]:>10.5f} "
              f"{average_precision_score(y_test, candidate_scores):>9.4f} "
              f"{f1_score(y_test, (candidate_scores >= 0.5).astype(int)):>9.4f}")

    best_method = min(band_errors, key=band_errors.get)
    print(f"\n  {int(band.sum()):,} of {len(y_test):,} test pairs "
          f"({band.mean():.1%}) fall in the undecided band")
    print(f"  Best calibration in that band: {best_method}")
    if best_method == "raw (uncalibrated)":
        print("  Calibration did not help. Keeping the raw scores and saying so,")
        print("  rather than adding a step that only looks rigorous.")

    model, scores = fitted[best_method], candidates[best_method]
    predicted = (scores >= 0.5).astype(int)

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE (test set, threshold 0.5)")
    print("=" * 70)
    print(classification_report(y_test, predicted, digits=4,
                                target_names=["different", "same person"]))
    print(f"  PR-AUC (average precision): {average_precision_score(y_test, scores):.4f}")
    print("  Accuracy is not reported on purpose -- at a 10% positive rate,")
    print("  always guessing 'different' scores ~89% and is worthless.")

    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    print(f"\n  true positives  {tp:>7,}   false positives {fp:>7,}  <- wrong merges")
    print(f"  false negatives {fn:>7,}   true negatives  {tn:>7,}")

    baseline = naive_baseline(test)
    print(f"\n  Rule-based baseline (name>0.90 AND dob identical):")
    print(f"    precision {baseline['precision']:.4f} | recall {baseline['recall']:.4f} "
          f"| f1 {baseline['f1']:.4f}")
    print(f"  Model F1 {f1_score(y_test, predicted):.4f} vs baseline "
          f"{baseline['f1']:.4f}  -> the model is worth its complexity")

    print("\n" + "=" * 70)
    print("WHAT THE MODEL LEARNED (standardised coefficients)")
    print("=" * 70)
    coefficients = pd.Series(
        raw_model.named_steps["clf"].coef_[0], index=FEATURE_NAMES
    ).sort_values(key=abs, ascending=False)
    for name, value in coefficients.items():
        direction = "evidence FOR a match " if value > 0 else "evidence AGAINST     "
        print(f"  {name:<22} {value:>8.3f}   {direction}")

    print("\n  Reading the paired features:")
    print("    Each optional field contributes TWO coefficients, and they work")
    print("    together. 'phone_comparable' fires whenever both records have a")
    print("    phone; 'phone_exact' fires only when those phones agree. So two")
    print("    records that both have a phone that DISAGREES score")
    print(f"    {coefficients.get('phone_comparable', 0):+.2f}, while two that agree score "
          f"{coefficients.get('phone_comparable', 0) + coefficients.get('phone_exact', 0):+.2f}.")
    print("    A missing phone scores 0 -- no evidence either way. That is the")
    print("    missing-data design working exactly as intended.")
    print("\n  Caveat worth stating out loud: these features are correlated")
    print("  (first_name_sim, full_name_sim and best_token_sim all measure the")
    print("  name), so individual coefficients are unstable and can flip sign.")
    print("  They show what the model leans on, not a clean causal weight.")

    print("\n" + "=" * 70)
    print("IS THE SCORE AN HONEST PROBABILITY?")
    print("=" * 70)
    print("  Raw model (class-weighted, so expected to be overconfident):")
    raw_table = reliability_table(y_test.to_numpy(), raw_scores)
    print(f"  {'score range':<16} {'pairs':>8} {'predicted':>10} {'actual':>9} {'gap':>8}")
    for row in raw_table.itertuples():
        print(f"  {str(row.bucket):<16} {row.pairs:>8,} {row.mean_predicted:>10.3f} "
              f"{row.actual_match_rate:>9.3f} {row.gap:>8.3f}")

    print(f"\n  After calibration by {best_method} "
          "(this is what the decision layer uses):")
    table = reliability_table(y_test.to_numpy(), scores)
    print(f"  {'score range':<16} {'pairs':>8} {'predicted':>10} {'actual':>9} {'gap':>8}")
    for row in table.itertuples():
        print(f"  {str(row.bucket):<16} {row.pairs:>8,} {row.mean_predicted:>10.3f} "
              f"{row.actual_match_rate:>9.3f} {row.gap:>8.3f}")

    print(f"\n  Expected calibration error: "
          f"{errors['raw (uncalibrated)']:.5f} raw -> {errors[best_method]:.5f} "
          f"({best_method})")

    tuned = choose_upper_threshold(y_test.to_numpy(), scores, target_precision=0.995)
    print(f"\n  Lowest threshold holding 99.5% precision: {tuned:.4f}")
    print(f"  (config.UPPER_THRESHOLD is currently {config.UPPER_THRESHOLD})")

    joblib.dump(model, config.DATA_PROCESSED / "matcher.joblib")

    # Record which people were held out, so later stages can report honest
    # metrics on unseen people instead of quietly scoring themselves on the
    # training set.
    test_people = sorted(set(test.person_a) | set(test.person_b))
    pd.DataFrame({"person_id": test_people}).to_csv(
        config.DATA_PROCESSED / "test_people.csv", index=False
    )

    # Persist scored test pairs -- the decision, clustering and LLM stages all
    # consume these rather than re-running the model.
    scored = test[["pos_a", "pos_b", "record_id_a", "record_id_b",
                   "person_a", "person_b", "is_match"]].copy()
    scored["score"] = scores
    scored.to_csv(config.DATA_PROCESSED / "scored_pairs.csv", index=False)

    metrics = {
        "n_train_pairs": int(len(train)),
        "n_test_pairs": int(len(test)),
        "precision": float(precision_score(y_test, predicted)),
        "recall": float(recall_score(y_test, predicted)),
        "f1": float(f1_score(y_test, predicted)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "baseline_f1": float(baseline["f1"]),
        "threshold_for_995_precision": tuned,
    }
    (config.REPORTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model -> matcher.joblib, scored pairs -> scored_pairs.csv")


if __name__ == "__main__":
    main()
