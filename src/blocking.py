"""
Step 3 of the pipeline: blocking, a.k.a. the reason this project scales.

With 31,195 records there are 31,195 x 31,194 / 2 = ~487 million possible
pairs. We cannot score all of them, and at 10 million records the number
becomes 50 trillion and the whole approach dies.

So instead of comparing everything, we only compare records that agree on at
least one cheap-to-compute key. Each key is a different bet about which field
survived the data entry damage:

    name_year   the name is mangled but phonetically close, and the year is right
    name_pin    same, for records with no usable date of birth
    phone_tail  the phone digits are right even if the formatting is not
    email_user  the email local part is right even if the domain changed
    dob_pin     the NAME is destroyed (nickname) but date and area are right

Using several keys and taking the union is the whole trick. Any single key
misses a large slice of the true duplicates; together they miss very few.

The two numbers that matter:
    reduction ratio   -- how much work did we avoid?
    pair completeness -- what fraction of the TRUE duplicate pairs survived?

These trade off against each other. Cheaper blocking loses recall you can
never get back, because a pair that is never generated can never be scored,
no matter how good the model downstream is.

Run:  python -m src.blocking
"""

import itertools
from collections import defaultdict

import pandas as pd

import config
from src.load import load_answer_key, load_standardized


def build_block_keys(frame: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    """Map each blocking strategy to {key value -> list of row positions}.

    A single record can land in several buckets of the same strategy -- a
    person with three name tokens produces three name_year keys. That is
    deliberate: it is what makes the keys survive swapped name order.
    """
    blocks: dict[str, dict[str, list[int]]] = {
        name: defaultdict(list) for name in config.BLOCKING_KEYS
    }

    for position, row in enumerate(frame.itertuples(index=False)):
        soundex_tokens = [t for t in (row.soundex_tokens or "").split() if t]
        birth_year = row.birth_year or ""
        pin3 = (row.pincode_clean or "")[:3]
        phone_tail = row.phone_tail or ""
        email_user = row.email_user or ""
        dob = row.dob_clean or ""

        if birth_year:
            # Require TWO name tokens to agree phonetically, not one.
            #
            # The single-token version of this key was measured generating 76%
            # of all candidate pairs while contributing only 107 true pairs
            # that no other key found -- a soundex code plus a birth year is
            # simply not selective enough on its own. Demanding a pair of
            # tokens keeps the key robust to one mangled name part while
            # cutting its cost by more than an order of magnitude.
            #
            # Records left with a single usable token (initials dropped by
            # abbreviation) fall back to the loose form, because for those
            # there is nothing more selective available.
            if len(soundex_tokens) >= 2:
                for first, second in itertools.combinations(soundex_tokens, 2):
                    blocks["name_year"][f"{first}+{second}|{birth_year}"].append(position)
            elif soundex_tokens:
                blocks["name_year"][f"{soundex_tokens[0]}|{birth_year}"].append(position)

        if pin3:
            for token in soundex_tokens:
                blocks["name_pin"][f"{token}|{pin3}"].append(position)

        if phone_tail:
            blocks["phone_tail"][phone_tail].append(position)

        # Very short local parts ('s', 'ab') are too generic to be a useful key.
        if len(email_user) >= 4:
            blocks["email_user"][email_user].append(position)

        if dob and pin3:
            blocks["dob_pin"][f"{dob}|{pin3}"].append(position)

    return blocks


def generate_candidate_pairs(
    blocks: dict[str, dict[str, list[int]]]
) -> dict[tuple[int, int], set[str]]:
    """Turn buckets into candidate pairs, remembering which keys found each one.

    Storing the source keys is not decoration. 'How many independent blocking
    keys agreed on this pair' turns out to be a genuinely predictive feature,
    and it also lets us report how much each key actually contributed.
    """
    pairs: dict[tuple[int, int], set[str]] = defaultdict(set)
    skipped_blocks = 0

    for key_name, buckets in blocks.items():
        for bucket_value, positions in buckets.items():
            if len(positions) < 2:
                continue
            # A bucket this large is not selective enough to be worth the
            # quadratic cost -- it is usually a data artefact rather than a
            # real cluster of the same person.
            if len(positions) > config.MAX_BLOCK_SIZE:
                skipped_blocks += 1
                continue
            for a, b in itertools.combinations(sorted(positions), 2):
                pairs[(a, b)].add(key_name)

    if skipped_blocks:
        print(f"  (skipped {skipped_blocks:,} oversized blocks "
              f"above {config.MAX_BLOCK_SIZE} records)")
    return pairs


def evaluate_blocking(
    pairs: dict[tuple[int, int], set[str]],
    frame: pd.DataFrame,
    answer_key: pd.DataFrame,
) -> pd.DataFrame:
    """Score the blocking stage against ground truth.

    This is the only place the answer key is allowed anywhere near the
    pipeline before a decision is made, and it is used purely to measure --
    never to select pairs.
    """
    truth = dict(zip(answer_key.record_id, answer_key.person_id))
    person_of_position = [truth.get(rid) for rid in frame.record_id]

    n_records = len(frame)
    total_possible = n_records * (n_records - 1) // 2

    # Every true duplicate pair that exists in the data, whether we found it or not.
    cluster_sizes = pd.Series(person_of_position).value_counts()
    total_true_pairs = int((cluster_sizes * (cluster_sizes - 1) // 2).sum())

    found_true = sum(
        1 for (a, b) in pairs
        if person_of_position[a] is not None
        and person_of_position[a] == person_of_position[b]
    )

    n_candidates = len(pairs)
    reduction = 1 - (n_candidates / total_possible)
    completeness = found_true / total_true_pairs if total_true_pairs else 0.0

    print("\n" + "=" * 70)
    print("BLOCKING PERFORMANCE")
    print("=" * 70)
    print(f"  Records                     {n_records:>14,}")
    print(f"  All possible pairs          {total_possible:>14,}")
    print(f"  Candidate pairs generated   {n_candidates:>14,}")
    print(f"  Reduction ratio             {reduction:>14.4%}  (work avoided)")
    print()
    print(f"  True duplicate pairs        {total_true_pairs:>14,}")
    print(f"  ...of which we kept         {found_true:>14,}")
    print(f"  Pair completeness           {completeness:>14.2%}  (recall ceiling)")
    print()
    print(f"  Positive rate in candidates {found_true / max(n_candidates, 1):>14.2%}"
          "  (this is why accuracy is a useless metric here)")

    # Per-key contribution: how many true pairs would we lose if we dropped
    # this key entirely? Keys with a high solo contribution are earning their cost.
    print("\n  Contribution by blocking key:")
    rows = []
    for key_name in config.BLOCKING_KEYS:
        subset = [(a, b) for (a, b), keys in pairs.items() if key_name in keys]
        true_here = sum(
            1 for (a, b) in subset
            if person_of_position[a] is not None
            and person_of_position[a] == person_of_position[b]
        )
        only_here = sum(
            1 for (a, b), keys in pairs.items()
            if keys == {key_name}
            and person_of_position[a] is not None
            and person_of_position[a] == person_of_position[b]
        )
        rows.append({
            "key": key_name,
            "candidates": len(subset),
            "true_pairs_found": true_here,
            "true_pairs_only_this_key": only_here,
            "precision": true_here / max(len(subset), 1),
        })
        print(f"    {key_name:<12} {len(subset):>9,} candidates | "
              f"{true_here:>7,} true | {only_here:>6,} found ONLY by this key | "
              f"precision {true_here / max(len(subset), 1):>6.1%}")
    print("=" * 70)

    return pd.DataFrame(rows)


def main() -> None:
    frame = load_standardized()
    answer_key = load_answer_key()
    print(f"Loaded {len(frame):,} standardized records")

    print("\nBuilding blocking keys...")
    blocks = build_block_keys(frame)
    for key_name, buckets in blocks.items():
        multi = sum(1 for v in buckets.values() if len(v) > 1)
        print(f"  {key_name:<12} {len(buckets):>8,} distinct values "
              f"({multi:,} with 2+ records)")

    print("\nGenerating candidate pairs...")
    pairs = generate_candidate_pairs(blocks)

    key_report = evaluate_blocking(pairs, frame, answer_key)
    key_report.to_csv(config.REPORTS / "blocking_by_key.csv", index=False)

    # Persist the candidate pairs for the feature stage. We store row positions
    # plus the record ids so the next stage never has to re-derive anything.
    record_ids = frame.record_id.tolist()
    pair_frame = pd.DataFrame(
        [
            {
                "pos_a": a,
                "pos_b": b,
                "record_id_a": record_ids[a],
                "record_id_b": record_ids[b],
                "blocking_keys": "|".join(sorted(keys)),
                "n_blocking_keys": len(keys),
            }
            for (a, b), keys in pairs.items()
        ]
    )
    path = config.DATA_PROCESSED / "candidate_pairs.csv"
    pair_frame.to_csv(path, index=False)
    print(f"\nWrote {len(pair_frame):,} candidate pairs -> {path.name}")


if __name__ == "__main__":
    main()
