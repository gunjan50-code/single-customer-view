"""
Step 1 of the pipeline: build a fictional bank's messy customer data.

We invent N_PEOPLE real human beings, then scatter damaged copies of them
across six source systems. The six CSVs written to data/raw/ are what the
rest of the pipeline sees -- they contain NO person_id. The answer key is
written separately and is only ever used for scoring, never for matching.

Run:  python -m src.generate_data
"""

import random
from datetime import date, timedelta

import pandas as pd

import config
from src import corrupt
from src import reference_data as ref

RECORD_COLUMNS = [
    "record_id", "source_system", "first_name", "middle_name", "last_name",
    "dob", "email", "phone", "address_line", "city", "state", "pincode",
    "created_date",
]


def make_person(person_id: int, rng: random.Random) -> dict:
    """Create one clean, canonical human being."""
    if rng.random() < 0.52:
        first = rng.choice(ref.FIRST_NAMES_MALE)
    else:
        first = rng.choice(ref.FIRST_NAMES_FEMALE)

    middle = rng.choice(ref.MIDDLE_NAMES)
    last = rng.choice(ref.LAST_NAMES)

    city = rng.choice(list(ref.CITY_VARIANTS.keys()))
    state = ref.CITY_TO_STATE.get(city, "")

    # Birthdays between 1955 and 2006, uniformly.
    dob = date(1955, 1, 1) + timedelta(days=rng.randrange(0, 365 * 51))

    # Email derived from the name, so the email_variant corruption has
    # something realistic to mangle.
    local = f"{first.lower()}.{last[0].lower()}"
    if rng.random() < 0.35:
        local = f"{first.lower()}{rng.randrange(1, 999)}"
    email = f"{local}@{rng.choice(ref.EMAIL_DOMAINS)}"

    phone_digits = rng.choice("6789") + "".join(str(rng.randrange(10)) for _ in range(9))

    address = f"{rng.randrange(1, 400)} {rng.choice(ref.STREET_NAMES)}"
    pincode = f"{rng.randrange(110, 700)}{rng.randrange(100, 999)}"

    return {
        "person_id": person_id,
        "first_name": first,
        "middle_name": middle,
        "last_name": last,
        "dob": dob,
        "email": email,
        "phone": phone_digits,
        "address_line": address,
        "city": city,
        "state": state,
        "pincode": pincode,
    }


def build_source_record(person: dict, system: str, record_id: str,
                        rng: random.Random) -> tuple[dict, list[str]]:
    """Produce one system's damaged view of a person."""
    damaged, applied = corrupt.corrupt_record(person, config.CORRUPTION_RATES, rng)

    # DOB and phone are stored as strings, and every system picked a different
    # format years ago. This is formatting, not damage -- but it breaks exact
    # matching just as thoroughly.
    dob_value = damaged.get("dob")
    if isinstance(dob_value, date):
        dob_str = corrupt.format_dob(dob_value, rng)
    else:
        dob_str = ""  # the missing_dob corruption fired

    phone_value = damaged.get("phone") or ""
    phone_str = corrupt.format_phone(phone_value, rng) if phone_value else ""

    # When the record was created, used later for survivorship ("most recent wins").
    created = date(2019, 1, 1) + timedelta(days=rng.randrange(0, 365 * 6))

    record = {
        "record_id": record_id,
        "source_system": system,
        "first_name": damaged.get("first_name", ""),
        "middle_name": damaged.get("middle_name", ""),
        "last_name": damaged.get("last_name", ""),
        "dob": dob_str,
        "email": damaged.get("email", ""),
        "phone": phone_str,
        "address_line": damaged.get("address_line", ""),
        "city": damaged.get("city", ""),
        "state": damaged.get("state", ""),
        "pincode": str(damaged.get("pincode", "")),
        "created_date": created.isoformat(),
    }
    return record, applied


def generate() -> None:
    rng = random.Random(config.RANDOM_SEED)

    people = [make_person(pid, rng) for pid in range(1, config.N_PEOPLE + 1)]
    print(f"Created {len(people):,} distinct real people")

    # Households: give some people the phone number of an earlier person, and
    # often their address too. These pairs look extremely similar on the
    # strongest features while being different human beings, so they are the
    # hardest negatives in the dataset -- and the reason the matcher cannot
    # simply trust a phone match.
    n_shared = 0
    for index in range(1, len(people)):
        if rng.random() < config.SHARED_PHONE_RATE:
            relative = people[rng.randrange(0, index)]
            people[index]["phone"] = relative["phone"]
            if rng.random() < 0.6:  # same household, so same address
                people[index]["address_line"] = relative["address_line"]
                people[index]["city"] = relative["city"]
                people[index]["state"] = relative["state"]
                people[index]["pincode"] = relative["pincode"]
                if rng.random() < 0.5:  # and often the same surname
                    people[index]["last_name"] = relative["last_name"]
            n_shared += 1
    print(f"  {n_shared:,} people share a phone number with a relative "
          f"(hard negatives)")

    records_by_system: dict[str, list[dict]] = {s: [] for s in config.SOURCE_SYSTEMS}
    answer_key: list[dict] = []
    counters = {s: 0 for s in config.SOURCE_SYSTEMS}

    system_names = list(config.SOURCE_SYSTEMS)
    system_weights = [config.SOURCE_SYSTEMS[s]["weight"] for s in system_names]
    spread_options = list(config.SPREAD_DISTRIBUTION)
    spread_weights = list(config.SPREAD_DISTRIBUTION.values())

    for person in people:
        # First decide how widely this person is scattered, then pick which
        # systems -- weighted, and without replacement.
        n_systems = rng.choices(spread_options, weights=spread_weights, k=1)[0]
        chosen: list[str] = []
        pool, pool_weights = list(system_names), list(system_weights)
        for _ in range(min(n_systems, len(pool))):
            pick = rng.choices(pool, weights=pool_weights, k=1)[0]
            index = pool.index(pick)
            pool.pop(index)
            pool_weights.pop(index)
            chosen.append(pick)

        for system in chosen:
            copies = 2 if rng.random() < config.WITHIN_SYSTEM_DUPLICATE_RATE else 1

            for _ in range(copies):
                counters[system] += 1
                record_id = f"{system[:3].upper()}-{counters[system]:06d}"
                record, applied = build_source_record(person, system, record_id, rng)
                records_by_system[system].append(record)
                answer_key.append({
                    "record_id": record_id,
                    "person_id": person["person_id"],
                    "corruptions": "|".join(applied),
                })

    total = 0
    for system, records in records_by_system.items():
        frame = pd.DataFrame(records, columns=RECORD_COLUMNS)
        path = config.DATA_RAW / f"{system}.csv"
        frame.to_csv(path, index=False)
        total += len(frame)
        print(f"  {system:<12} {len(frame):>7,} records -> {path.name}")

    key_frame = pd.DataFrame(answer_key)
    key_frame.to_csv(config.DATA_RAW / "_answer_key.csv", index=False)

    # How many people ended up with more than one record? That is the
    # duplication rate the engine has to discover on its own.
    per_person = key_frame.groupby("person_id").size()
    duplicated_people = int((per_person > 1).sum())
    represented = len(per_person)

    print(f"\n{total:,} raw records describing {represented:,} real people")
    print(f"Duplication rate: {1 - represented / total:.1%} of records are redundant")
    print(f"{duplicated_people:,} people appear more than once "
          f"({duplicated_people / represented:.1%})")
    print(f"Largest single person spans {per_person.max()} records")
    print("\nAnswer key written to data/raw/_answer_key.csv (scoring only -- "
          "never used for matching)")


if __name__ == "__main__":
    generate()
