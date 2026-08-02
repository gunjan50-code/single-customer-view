"""
The corruption engine.

This is the heart of the project's evaluation strategy. We start from clean,
known people and deliberately damage copies of them in the ways real data
entry actually fails. Because we control the damage, we know the correct
answer for every record -- which gives us labelled training data and an
honest test set without paying a human to label anything.

In a real engagement, labelled duplicate pairs are the scarcest resource on
the project. Generating ground truth this way is a standard technique for
building and validating a matcher before client labels exist.

Every function takes a record dict and returns a NEW dict. Nothing mutates
in place, so a single clean person can spawn many independent messy copies.
"""

import random
from datetime import date

from src import reference_data as ref


# --------------------------------------------------------------- name damage

def keyboard_typo(record: dict, rng: random.Random) -> dict:
    """Replace one letter with a physically adjacent key.

    Real typos are not random. Someone typing 'Sharma' in a hurry hits the key
    next to the one they meant, producing 'Sharna' or 'Shatma'. Random
    character substitution would be far easier for a matcher to catch, and
    would make our results look better than they deserve to.
    """
    out = dict(record)
    field = rng.choice(["first_name", "last_name"])
    word = out.get(field) or ""
    if len(word) < 3:
        return out

    # Never damage the first letter -- it survives in most real typos, and
    # our blocking keys depend on it.
    positions = [i for i in range(1, len(word)) if word[i].lower() in ref.KEYBOARD_NEIGHBOURS]
    if not positions:
        return out

    pos = rng.choice(positions)
    neighbours = ref.KEYBOARD_NEIGHBOURS[word[pos].lower()]
    replacement = rng.choice(neighbours)
    if word[pos].isupper():
        replacement = replacement.upper()

    out[field] = word[:pos] + replacement + word[pos + 1:]
    return out


def abbreviate_name(record: dict, rng: random.Random) -> dict:
    """'Rajesh Kumar Sharma' -> 'R. Kumar Sharma' or 'R. K. Sharma'.

    Extremely common in billing and legacy systems with short field limits.
    """
    out = dict(record)
    if out.get("first_name"):
        out["first_name"] = f"{out['first_name'][0]}."
    if rng.random() < 0.6 and out.get("middle_name"):
        out["middle_name"] = f"{out['middle_name'][0]}."
    return out


def swap_name_order(record: dict, rng: random.Random) -> dict:
    """'Rajesh Sharma' -> 'Sharma Rajesh'.

    Happens constantly in South Indian records where surname-first is the
    norm, and whenever a form labels its fields ambiguously.
    """
    out = dict(record)
    out["first_name"], out["last_name"] = out.get("last_name"), out.get("first_name")
    return out


def apply_nickname(record: dict, rng: random.Random) -> dict:
    """'Rajesh' -> 'Raju'. Typically how the support system records a caller."""
    out = dict(record)
    nick = ref.NICKNAMES.get(out.get("first_name"))
    if nick:
        out["first_name"] = nick
    return out


def drop_middle_name(record: dict, rng: random.Random) -> dict:
    """Many systems simply have no middle name field."""
    out = dict(record)
    out["middle_name"] = ""
    return out


# ------------------------------------------------------------ address damage

def city_variant(record: dict, rng: random.Random) -> dict:
    """'Bengaluru' -> 'Bangalore' / 'Blore' / 'BLR'."""
    out = dict(record)
    variants = ref.CITY_VARIANTS.get(out.get("city"))
    if variants:
        out["city"] = rng.choice(variants)
    return out


def address_abbreviation(record: dict, rng: random.Random) -> dict:
    """'12 MG Road' -> '12 MG Rd'."""
    out = dict(record)
    address = out.get("address_line") or ""
    for long_form, short_form in ref.ADDRESS_ABBREVIATIONS.items():
        if long_form in address and rng.random() < 0.7:
            address = address.replace(long_form, short_form)
    # Systems disagree on how to write a house number.
    if rng.random() < 0.4:
        address = rng.choice(["#", "No. ", "No ", ""]) + address
    out["address_line"] = address
    return out


def relocate(record: dict, rng: random.Random) -> dict:
    """The customer moved house between one system's record and another's.

    This is not data entry damage -- it is the world changing. It matters
    enormously because address is otherwise an almost perfect matching
    signal, and a generator where nobody ever moves produces a problem far
    easier than the real one. Roughly one record in eight carries a stale
    address, which is consistent with real customer files spanning years.

    It also creates the genuinely hard cases: two records, same person,
    different city, different pincode, different street. Only the name, date
    of birth and contact details can save those.
    """
    out = dict(record)
    new_city = rng.choice(list(ref.CITY_VARIANTS.keys()))
    out["city"] = new_city
    out["state"] = ref.CITY_TO_STATE.get(new_city, "")
    out["address_line"] = f"{rng.randrange(1, 400)} {rng.choice(ref.STREET_NAMES)}"
    out["pincode"] = f"{rng.randrange(110, 700)}{rng.randrange(100, 999)}"
    return out


def pincode_damage(record: dict, rng: random.Random) -> dict:
    """Drop a digit, add a space, or transpose two digits."""
    out = dict(record)
    pin = str(out.get("pincode") or "")
    if len(pin) != 6:
        return out

    mode = rng.choice(["drop", "space", "transpose"])
    if mode == "drop":
        cut = rng.randrange(6)
        out["pincode"] = pin[:cut] + pin[cut + 1:]
    elif mode == "space":
        out["pincode"] = pin[:3] + " " + pin[3:]
    else:
        i = rng.randrange(5)
        out["pincode"] = pin[:i] + pin[i + 1] + pin[i] + pin[i + 2:]
    return out


# -------------------------------------------------------- contact detail damage

def email_variant(record: dict, rng: random.Random) -> dict:
    """A real person's several email addresses for the same identity.

    Note the Gmail dot case: 'rajesh.k@gmail.com' and 'rajeshk@gmail.com'
    are literally the same inbox, because Gmail ignores dots. A naive exact
    match treats them as different people. standardize.py fixes this.
    """
    out = dict(record)
    email = out.get("email") or ""
    if "@" not in email:
        return out

    local, domain = email.split("@", 1)
    mode = rng.choice(["dots", "number", "domain", "shorten"])

    if mode == "dots":
        local = local.replace(".", "") if "." in local else local
    elif mode == "number":
        local = local + str(rng.randrange(1, 99))
    elif mode == "domain":
        domain = rng.choice(ref.EMAIL_DOMAINS)
    elif mode == "shorten" and len(local) > 4:
        local = local[:4]

    out["email"] = f"{local}@{domain}"
    return out


def phone_typo(record: dict, rng: random.Random) -> dict:
    """Mistype a single digit of the phone number.

    Without this, a phone number that survives to both records is a perfect
    oracle, and the model learns to lean on it almost exclusively. Real phone
    numbers get transposed and fat-fingered like everything else, and a
    matcher that collapses when the phone is wrong is not much of a matcher.
    """
    out = dict(record)
    digits = out.get("phone") or ""
    if len(digits) < 10:
        return out
    position = rng.randrange(1, len(digits))
    if rng.random() < 0.5 and position < len(digits) - 1:
        # transposition: 9977 -> 9797
        chars = list(digits)
        chars[position], chars[position + 1] = chars[position + 1], chars[position]
        out["phone"] = "".join(chars)
    else:
        replacement = str((int(digits[position]) + rng.choice([1, -1, 2])) % 10)
        out["phone"] = digits[:position] + replacement + digits[position + 1:]
    return out


def format_phone(digits: str, rng: random.Random) -> str:
    """Render the same 10 digits the way six different systems would store it."""
    style = rng.choice(
        ["plain", "plus91_space", "zero_dash", "plus91_solid", "space", "brackets"]
    )
    if style == "plain":
        return digits
    if style == "plus91_space":
        return f"+91 {digits[:5]} {digits[5:]}"
    if style == "zero_dash":
        return f"0{digits[:5]}-{digits[5:]}"
    if style == "plus91_solid":
        return f"+91{digits}"
    if style == "space":
        return f"{digits[:5]} {digits[5:]}"
    return f"({digits[:4]}) {digits[4:]}"


def format_dob(dob: date, rng: random.Random) -> str:
    """Same birthday, six different string formats.

    '%m/%d/%Y' is included on purpose: it is genuinely ambiguous against
    '%d/%m/%Y' for any day under 13, which is a real and nasty source of
    silent data corruption in Indian systems built on US software.
    """
    fmt = rng.choice(["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y", "%d.%m.%Y", "%m/%d/%Y"])
    return dob.strftime(fmt)


# ------------------------------------------------------------- generic damage

def missing_field(record: dict, field: str) -> dict:
    """Blank a field out entirely. Optional fields are skipped constantly."""
    out = dict(record)
    out[field] = ""
    return out


def extra_whitespace(record: dict, rng: random.Random) -> dict:
    """Leading, trailing, and doubled spaces from copy-paste."""
    out = dict(record)
    field = rng.choice(["first_name", "last_name", "address_line", "city"])
    value = out.get(field) or ""
    if value:
        out[field] = rng.choice([f"  {value}", f"{value}  ", value.replace(" ", "  ")])
    return out


def case_change(record: dict, rng: random.Random) -> dict:
    """ALL CAPS from a mainframe, or all lowercase from a web form."""
    out = dict(record)
    transform = rng.choice([str.upper, str.lower])
    for field in ("first_name", "middle_name", "last_name", "city", "address_line"):
        if out.get(field) and rng.random() < 0.8:
            out[field] = transform(out[field])
    return out


# ------------------------------------------------------------------ driver

# Maps the rate keys in config.CORRUPTION_RATES to the function that applies them.
# Order matters: relocate runs first so that city_variant and
# address_abbreviation then damage the *new* address rather than the old one.
CORRUPTORS = {
    "relocate": relocate,
    "keyboard_typo": keyboard_typo,
    "phone_typo": phone_typo,
    "abbreviate_name": abbreviate_name,
    "swap_name_order": swap_name_order,
    "nickname": apply_nickname,
    "drop_middle_name": drop_middle_name,
    "city_variant": city_variant,
    "address_abbreviation": address_abbreviation,
    "pincode_damage": pincode_damage,
    "email_variant": email_variant,
    "extra_whitespace": extra_whitespace,
    "case_change": case_change,
}

MISSING_FIELD_RATES = {
    "missing_email": "email",
    "missing_phone": "phone",
    "missing_dob": "dob",
}


def corrupt_record(record: dict, rates: dict, rng: random.Random) -> tuple[dict, list[str]]:
    """Apply every corruption independently at its configured rate.

    Returns the damaged record plus the list of damage actually applied, so
    the generator can write an audit trail. Being able to say "this pair was
    hard because it had a nickname AND a swapped name order" is genuinely
    useful when you are debugging why the matcher missed something.
    """
    out = dict(record)
    applied: list[str] = []

    for name, fn in CORRUPTORS.items():
        if rng.random() < rates.get(name, 0.0):
            candidate = fn(out, rng)
            # Only log damage that actually changed something. Several
            # corruptors are legitimately no-ops -- there is no nickname for
            # "Prakash", and a two-letter name has no safe position to typo.
            # Logging those anyway would make the audit trail lie, and we use
            # it later to explain which damage the matcher struggles with.
            if candidate != out:
                out = candidate
                applied.append(name)

    for rate_key, field in MISSING_FIELD_RATES.items():
        if rng.random() < rates.get(rate_key, 0.0) and out.get(field):
            out = missing_field(out, field)
            applied.append(rate_key)

    return out, applied
