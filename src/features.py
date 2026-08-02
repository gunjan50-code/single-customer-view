"""
Step 4 of the pipeline: turn each candidate pair into a row of numbers.

This is the conceptual pivot of the whole project. Matching two messy records
feels like a fuzzy, judgement-based problem. But once every pair is described
by a fixed vector of similarity scores, it becomes an ordinary supervised
classification problem that scikit-learn can solve in three lines.

Design note on missing data. When one record has no email, the email
similarity is not zero -- it is *unknown*, which is a different thing. Scoring
it as zero would teach the model that a missing email is evidence against a
match, which is wrong. So every field that can be absent gets two columns:
the similarity (0 when uncomparable) and a `_comparable` flag saying whether
the similarity means anything at all. The model can then learn "high email
similarity is strong evidence, but no email is no evidence".

Run:  python -m src.features
"""

import pandas as pd
from rapidfuzz import fuzz

import config
from src.load import load_answer_key, load_standardized

# Columns pulled out of the dataframe once, into plain Python lists.
# Indexing a list by position is dramatically faster than .iloc inside a loop
# over 170k pairs, and this stage is the hot loop of the pipeline.
NEEDED_COLUMNS = [
    "record_id", "source_system", "first_name_clean", "last_name_clean",
    "name_tokens", "soundex_tokens", "dob_clean", "birth_year",
    "email_clean", "email_user", "phone_clean", "address_clean",
    "city_clean", "pincode_clean",
]

FEATURE_NAMES = [
    "first_name_sim",
    "last_name_sim",
    "full_name_sim",
    "name_token_jaccard",
    "best_token_sim",
    "soundex_overlap",
    "dob_exact",
    "birth_year_match",
    "dob_comparable",
    "email_sim",
    "email_domain_match",
    "email_comparable",
    "phone_exact",
    "phone_comparable",
    "address_sim",
    "city_match",
    "pincode_match",
    "n_blocking_keys",
    "same_source_system",
    "n_missing_fields",
]


def ratio(a: str, b: str) -> float:
    """rapidfuzz similarity, rescaled to 0-1. Returns 0 if either side is empty."""
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def token_sort(a: str, b: str) -> float:
    """Similarity that ignores word order -- 'sneha naidu' vs 'naidu sneha'."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def jaccard(a: str, b: str) -> float:
    """Overlap of two space-separated token sets."""
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def best_token_similarity(a: str, b: str) -> float:
    """Similarity of the single best-matching token pair.

    Catches the case where one name part survived intact and the rest is
    wreckage -- 'S. R. Nsidu' still shares a strong 'naidu'/'nsidu' token
    with the clean record.
    """
    tokens_a = [t for t in a.split() if t]
    tokens_b = [t for t in b.split() if t]
    if not tokens_a or not tokens_b:
        return 0.0
    return max(fuzz.ratio(x, y) for x in tokens_a for y in tokens_b) / 100.0


def count_missing(*values: str) -> int:
    return sum(1 for v in values if not v)


def build_features(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    columns = {name: frame[name].fillna("").astype(str).tolist() for name in NEEDED_COLUMNS}

    first_name = columns["first_name_clean"]
    last_name = columns["last_name_clean"]
    name_tokens = columns["name_tokens"]
    soundex = columns["soundex_tokens"]
    dob = columns["dob_clean"]
    year = columns["birth_year"]
    email = columns["email_clean"]
    email_user = columns["email_user"]
    phone = columns["phone_clean"]
    address = columns["address_clean"]
    city = columns["city_clean"]
    pincode = columns["pincode_clean"]
    system = columns["source_system"]

    rows = []
    positions_a = pairs.pos_a.tolist()
    positions_b = pairs.pos_b.tolist()
    n_keys = pairs.n_blocking_keys.tolist()

    for i, (a, b) in enumerate(zip(positions_a, positions_b)):
        email_a, email_b = email[a], email[b]
        phone_a, phone_b = phone[a], phone[b]
        dob_a, dob_b = dob[a], dob[b]

        email_comparable = bool(email_a and email_b)
        phone_comparable = bool(phone_a and phone_b)
        dob_comparable = bool(dob_a and dob_b)

        domain_a = email_a.split("@")[-1] if email_comparable else ""
        domain_b = email_b.split("@")[-1] if email_comparable else ""

        rows.append((
            ratio(first_name[a], first_name[b]),
            ratio(last_name[a], last_name[b]),
            token_sort(name_tokens[a], name_tokens[b]),
            jaccard(name_tokens[a], name_tokens[b]),
            best_token_similarity(name_tokens[a], name_tokens[b]),
            jaccard(soundex[a], soundex[b]),
            float(dob_comparable and dob_a == dob_b),
            float(bool(year[a]) and year[a] == year[b]),
            float(dob_comparable),
            ratio(email_user[a], email_user[b]) if email_comparable else 0.0,
            float(email_comparable and domain_a == domain_b),
            float(email_comparable),
            float(phone_comparable and phone_a == phone_b),
            float(phone_comparable),
            token_sort(address[a], address[b]),
            float(bool(city[a]) and city[a] == city[b]),
            float(bool(pincode[a]) and pincode[a] == pincode[b]),
            float(n_keys[i]),
            float(system[a] == system[b]),
            float(count_missing(email_a, phone_a, dob_a)
                  + count_missing(email_b, phone_b, dob_b)),
        ))

    features = pd.DataFrame(rows, columns=FEATURE_NAMES)
    features.insert(0, "record_id_b", pairs.record_id_b.values)
    features.insert(0, "record_id_a", pairs.record_id_a.values)
    features.insert(0, "pos_b", positions_b)
    features.insert(0, "pos_a", positions_a)
    return features


def attach_labels(features: pd.DataFrame, frame: pd.DataFrame,
                  answer_key: pd.DataFrame) -> pd.DataFrame:
    """Add the ground-truth label and the person id of each side.

    person_id is carried through so the next stage can split train/test BY
    PERSON. Splitting randomly by pair would leak: the same person would
    appear on both sides of the split and the test score would be inflated.
    """
    truth = dict(zip(answer_key.record_id, answer_key.person_id))
    person_of_position = [truth.get(rid) for rid in frame.record_id]

    out = features.copy()
    out["person_a"] = [person_of_position[p] for p in out.pos_a]
    out["person_b"] = [person_of_position[p] for p in out.pos_b]
    out["is_match"] = (out.person_a == out.person_b).astype(int)
    return out


def main() -> None:
    frame = load_standardized()
    answer_key = load_answer_key()
    pairs = pd.read_csv(config.DATA_PROCESSED / "candidate_pairs.csv")
    print(f"Building features for {len(pairs):,} candidate pairs...")

    features = build_features(frame, pairs)
    labelled = attach_labels(features, frame, answer_key)

    path = config.DATA_PROCESSED / "pair_features.csv"
    labelled.to_csv(path, index=False)

    n_match = int(labelled.is_match.sum())
    print(f"\nWrote {len(labelled):,} labelled pairs -> {path.name}")
    print(f"  matches     {n_match:>9,} ({n_match / len(labelled):.2%})")
    print(f"  non-matches {len(labelled) - n_match:>9,}")

    print("\nMean feature value, matches vs non-matches:")
    print(f"  {'feature':<22} {'match':>8} {'non-match':>10} {'gap':>8}")
    means_match = labelled[labelled.is_match == 1][FEATURE_NAMES].mean()
    means_non = labelled[labelled.is_match == 0][FEATURE_NAMES].mean()
    gaps = (means_match - means_non).abs().sort_values(ascending=False)
    for name in gaps.index:
        print(f"  {name:<22} {means_match[name]:>8.3f} {means_non[name]:>10.3f} "
              f"{means_match[name] - means_non[name]:>8.3f}")


if __name__ == "__main__":
    main()
