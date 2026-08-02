"""
Step 2 of the pipeline: make every record look the same shape.

This stage does no matching at all. It just removes differences that are
purely cosmetic -- casing, punctuation, spacing, abbreviations, phone
formatting, date formatting. In a real project this unglamorous step
resolves a large share of the duplicates on its own, before any model runs,
and it is the cheapest win available.

It also builds the derived columns that blocking depends on.

Run:  python -m src.standardize
"""

import re

import jellyfish
import pandas as pd
from dateutil import parser as date_parser

import config
from src import reference_data as ref
from src.load import load_all_raw

PUNCTUATION = re.compile(r"[^\w\s]")
MULTISPACE = re.compile(r"\s+")
NON_DIGIT = re.compile(r"\D")


# ------------------------------------------------------------------ text

def clean_text(value: str) -> str:
    """Lowercase, strip punctuation, collapse repeated spaces."""
    if not value:
        return ""
    value = str(value).lower()
    value = PUNCTUATION.sub(" ", value)
    value = MULTISPACE.sub(" ", value)
    return value.strip()


def normalise_address(value: str) -> str:
    """Expand the street-type abbreviations that systems apply inconsistently.

    '4 Crs Rd' and '4 Cross Road' are the same address; without this they
    share almost no characters and every string metric scores them apart.
    """
    cleaned = clean_text(value)
    if not cleaned:
        return ""
    tokens = [ref.ADDRESS_EXPANSIONS.get(tok, tok) for tok in cleaned.split()]
    return " ".join(tokens)


def canonical_city(value: str) -> str:
    """'Blore' / 'Bangalore' / 'BLR' -> 'bengaluru'."""
    cleaned = clean_text(value)
    if not cleaned:
        return ""
    return ref.CITY_CANONICAL.get(cleaned, cleaned).lower()


# --------------------------------------------------------------- contact

def normalise_phone(value: str) -> str:
    """Strip every non-digit, drop country/trunk prefixes, keep the last 10.

    '+91 99774 40343', '099774-40343' and '(9977) 440343' all collapse to
    '9977440343'.
    """
    digits = NON_DIGIT.sub("", str(value or ""))
    if len(digits) < 10:
        return ""
    return digits[-10:]


def normalise_email(value: str) -> str:
    """Lowercase, and remove dots from the local part of Gmail addresses.

    Gmail genuinely ignores dots: 'rajesh.k@gmail.com' and 'rajeshk@gmail.com'
    deliver to the same inbox and belong to the same human. Treating them as
    different is a real and common source of duplicate customers.
    """
    value = str(value or "").strip().lower()
    if "@" not in value:
        return ""
    local, domain = value.rsplit("@", 1)
    if domain in {"gmail.com", "googlemail.com"}:
        local = local.replace(".", "")
    local = local.split("+", 1)[0]  # gmail also ignores everything after '+'
    return f"{local}@{domain}"


def normalise_pincode(value: str) -> str:
    """Digits only. Damaged 5-digit pincodes are kept -- they still carry signal."""
    return NON_DIGIT.sub("", str(value or ""))


# ------------------------------------------------------------------ date

def parse_dob(value: str) -> tuple[str, str]:
    """Parse any of our six date formats. Returns (iso_date, birth_year).

    Important caveat, and a genuinely realistic one: '11/06/1998' is
    ambiguous. It is 11 June under Indian convention and 6 November under US
    convention, and both appear in our sources because the fictional bank
    runs US-built software. We assume day-first, which is right most of the
    time and wrong for any day under 13.

    The year, however, is never ambiguous. That is exactly why blocking uses
    birth_year rather than the full date -- the part of the field that
    survives format confusion is the part worth building a key on.
    """
    text = str(value or "").strip()
    if not text:
        return "", ""
    try:
        parsed = date_parser.parse(text, dayfirst=True)
    except (ValueError, OverflowError):
        return "", ""
    return parsed.date().isoformat(), str(parsed.year)


# ------------------------------------------------------------ derived keys

def safe_soundex(value: str) -> str:
    """Soundex, guarding against empty or non-alphabetic input."""
    letters = re.sub(r"[^a-z]", "", str(value or "").lower())
    if not letters:
        return ""
    try:
        return jellyfish.soundex(letters)
    except Exception:
        return ""


def name_tokens(row: pd.Series) -> list[str]:
    """All name parts as a set, ignoring which field they landed in.

    This is how we survive swapped name order. 'Sneha Naidu' and
    'Naidu Sneha' produce the same token set, so any key built from tokens
    rather than from the first_name column will still bring them together.
    """
    parts = []
    for field in ("first_name_clean", "middle_name_clean", "last_name_clean"):
        value = row.get(field) or ""
        parts.extend(p for p in value.split() if len(p) > 1)  # drop 'r.' initials
    return sorted(set(parts))


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    for field in ("first_name", "middle_name", "last_name"):
        out[f"{field}_clean"] = out[field].map(clean_text)

    out["address_clean"] = out["address_line"].map(normalise_address)
    out["city_clean"] = out["city"].map(canonical_city)
    out["phone_clean"] = out["phone"].map(normalise_phone)
    out["email_clean"] = out["email"].map(normalise_email)
    out["pincode_clean"] = out["pincode"].map(normalise_pincode)

    parsed = out["dob"].map(parse_dob)
    out["dob_clean"] = [p[0] for p in parsed]
    out["birth_year"] = [p[1] for p in parsed]

    # Order-independent view of the name, used by both blocking and features.
    tokens = out.apply(name_tokens, axis=1)
    out["name_tokens"] = tokens.map(lambda t: " ".join(t))
    out["name_sorted"] = out["name_tokens"]  # readable alias for the features stage

    # Derived blocking helpers.
    out["email_user"] = out["email_clean"].map(lambda e: e.split("@")[0] if e else "")
    out["phone_tail"] = out["phone_clean"].map(lambda p: p[-6:] if p else "")
    out["soundex_tokens"] = tokens.map(
        lambda toks: " ".join(sorted({safe_soundex(t) for t in toks if t}))
    )

    return out


def main() -> None:
    raw = load_all_raw()
    print(f"Loaded {len(raw):,} raw records from {raw.source_system.nunique()} systems")

    cleaned = standardize(raw)

    path = config.DATA_PROCESSED / "standardized.csv"
    cleaned.to_csv(path, index=False)

    # A short report on what the cleaning step alone achieved.
    print(f"\nWrote {len(cleaned):,} standardized records -> {path.name}")
    print("\nField completeness after cleaning:")
    for field in ("email_clean", "phone_clean", "dob_clean", "pincode_clean", "city_clean"):
        filled = (cleaned[field].astype(str).str.len() > 0).mean()
        print(f"  {field:<16} {filled:>6.1%} populated")

    print("\nCollapse achieved by standardization alone:")
    for raw_field, clean_field in [("city", "city_clean"),
                                   ("phone", "phone_clean"),
                                   ("email", "email_clean")]:
        before = raw[raw_field].nunique()
        after = cleaned[clean_field].nunique()
        print(f"  {raw_field:<8} {before:>7,} distinct -> {after:>7,} distinct")


if __name__ == "__main__":
    main()
