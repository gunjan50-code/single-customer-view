"""
A demo helper: print a few real people as they appear across the six systems.

This is the single most useful screenshot for the README and for explaining
the project out loud -- it makes the problem obvious in five seconds without
any explanation of entity resolution at all.

Run:  python -m src.inspect_sample
"""

import pandas as pd

import config
from src.load import load_all_raw

DISPLAY_COLUMNS = [
    "source_system", "first_name", "middle_name", "last_name",
    "dob", "email", "phone", "address_line", "city", "pincode",
]


def main(n_people: int = 3) -> None:
    records = load_all_raw()
    key = pd.read_csv(config.DATA_RAW / "_answer_key.csv")
    merged = records.merge(key, on="record_id")

    counts = merged.groupby("person_id").size().sort_values(ascending=False)
    # Skip the very largest -- pick from the middle of the pack so the example
    # is typical rather than cherry-picked.
    candidates = counts[counts.between(3, 5)].index[:n_people]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 28)

    for person_id in candidates:
        subset = merged[merged.person_id == person_id]
        print("=" * 120)
        print(f"ONE REAL PERSON  (person_id={person_id})  -> {len(subset)} records "
              f"the bank currently believes are {len(subset)} different customers")
        print("=" * 120)
        print(subset[DISPLAY_COLUMNS].to_string(index=False))
        damage = sorted({c for row in subset["corruptions"].fillna("")
                         for c in row.split("|") if c})
        print(f"\nDamage present in this cluster: {', '.join(damage)}\n")


if __name__ == "__main__":
    main()
