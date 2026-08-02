"""
Central configuration for the single customer view pipeline.

Everything tunable lives here so the pipeline scripts stay readable and you
never have to hunt for a magic number.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DB_PATH = DATA_PROCESSED / "golden.db"

for _p in (DATA_RAW, DATA_PROCESSED, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data generation
RANDOM_SEED = 42

# How many real, distinct people exist in our fictional world.
N_PEOPLE = 20_000

# The six systems our fictional bank runs. Each one holds a messy subset of
# the same customers.
#   trust  -> used for survivorship: when two records disagree on a field,
#             the higher-trust source wins.
#   weight -> how likely this system is to hold any given customer. The CRM
#             sees almost everyone; support only sees people who called in.
SOURCE_SYSTEMS = {
    "web_signup": {"trust": 5, "weight": 0.20},  # customer typed it themselves
    "crm": {"trust": 4, "weight": 0.26},
    "billing": {"trust": 4, "weight": 0.20},
    "loyalty": {"trust": 3, "weight": 0.12},
    "branch": {"trust": 2, "weight": 0.14},  # clerk typed it by hand
    "support": {"trust": 1, "weight": 0.08},  # agent typed it during a call
}

# How many systems a single customer appears in.
# Deliberately long-tailed: most customers touched the bank once and live in
# one system. A minority of long-standing customers are scattered everywhere,
# and those are the expensive ones. A uniform distribution here would make
# almost every record a duplicate, which would both flatter the model and
# teach it the wrong prior.
SPREAD_DISTRIBUTION = {1: 0.70, 2: 0.18, 3: 0.07, 4: 0.03, 5: 0.015, 6: 0.005}

# Chance that a system holds the same customer twice -- someone re-registered
# instead of logging in. Within-system duplicates are the ones clients find
# most embarrassing.
WITHIN_SYSTEM_DUPLICATE_RATE = 0.04

# Fraction of people who share their phone number with another person -- a
# household landline, a family mobile, a small shop's number. These create
# genuine FALSE positives on the single strongest feature, which is the point:
# a matcher that treats "same phone" as proof will merge a father and son.
SHARED_PHONE_RATE = 0.08

# Probability that any given copy of a person gets each kind of damage.
# Tuned so records are realistically messy without being unmatchable.
CORRUPTION_RATES = {
    # Not damage -- the customer genuinely moved house. Without this, address
    # becomes an almost perfect matching signal and the problem is far easier
    # than the real one.
    "relocate": 0.13,
    "keyboard_typo": 0.25,
    # Stops the phone number from being a perfect oracle. Without it the model
    # leans on phone_exact almost exclusively and learns nothing else.
    "phone_typo": 0.16,
    "abbreviate_name": 0.20,
    "swap_name_order": 0.12,
    "nickname": 0.15,
    "drop_middle_name": 0.35,
    "city_variant": 0.30,
    "address_abbreviation": 0.40,
    "pincode_damage": 0.10,
    "missing_email": 0.18,
    "missing_phone": 0.15,
    "missing_dob": 0.12,
    "email_variant": 0.22,
    "extra_whitespace": 0.20,
    "case_change": 0.25,
}

# ---------------------------------------------------------------- blocking
# A pair of records is only compared if they agree on at least one of these
# cheap keys. This is what turns ~billions of comparisons into a few hundred
# thousand. See src/blocking.py.
#   name_year  -> phonetic name token + birth year
#   name_pin   -> phonetic name token + first 3 pincode digits
#   phone_tail -> last 6 digits of the phone number
#   email_user -> the local part of the email address
#   dob_pin    -> birth date + pincode region, with NO name component. This one
#                 exists specifically to catch nicknames: when 'Rajesh' is
#                 recorded as 'Raju', every name-based key fails, so we need at
#                 least one route that ignores the name entirely.
BLOCKING_KEYS = ["name_year", "name_pin", "phone_tail", "email_user", "dob_pin"]

# Safety valve: if a single block contains more records than this, it is too
# generic to be useful (e.g. everyone with a missing surname) and is skipped.
MAX_BLOCK_SIZE = 120

# ---------------------------------------------------------------- model
TEST_SIZE = 0.30  # split is done BY PERSON, not by pair, to avoid leakage

# The two thresholds that create the three-way decision.
# Tuned for high precision: a wrong merge is far worse than a missed duplicate.
UPPER_THRESHOLD = 0.95  # above this -> merge automatically
LOWER_THRESHOLD = 0.30  # below this -> reject automatically
# between the two -> send to a human (the "clerical review" band)

# ---------------------------------------------------------------- clustering
# Guards against "runaway clusters", where one bad link chains hundreds of
# unrelated people into a single blob.
MAX_CLUSTER_SIZE = 25
MIN_CLUSTER_COHESION = 0.55  # average pairwise score inside a cluster

# ---------------------------------------------------------------- llm
LLM_MODEL = "claude-sonnet-5"
LLM_MAX_PAIRS = 300  # cap so a demo run stays cheap

# Published per-million-token prices, used for the cost comparison table.
LLM_INPUT_COST_PER_MTOK = 3.00
LLM_OUTPUT_COST_PER_MTOK = 15.00
USD_TO_INR = 88.0
