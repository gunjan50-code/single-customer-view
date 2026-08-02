"""
Loading helpers shared by every stage of the pipeline.

Keeping this in one place means the "six separate systems" fiction stays
honest -- nothing downstream is allowed to peek at the answer key unless it
explicitly asks for it via load_answer_key().
"""

import pandas as pd

import config

# Everything is read as string. Pincodes have leading zeros, phone numbers are
# not numbers, and letting pandas guess dtypes here silently destroys data --
# which is itself a nice thing to be able to explain in an interview.
DTYPE = str


def load_all_raw() -> pd.DataFrame:
    """Concatenate the six source systems into one frame of dirty records."""
    frames = []
    for system in config.SOURCE_SYSTEMS:
        path = config.DATA_RAW / f"{system}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.generate_data` first."
            )
        frames.append(pd.read_csv(path, dtype=DTYPE, keep_default_na=False))

    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_answer_key() -> pd.DataFrame:
    """record_id -> person_id. Scoring only. Never joined in before a decision."""
    path = config.DATA_RAW / "_answer_key.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.generate_data` first."
        )
    return pd.read_csv(path, dtype={"record_id": str, "person_id": int,
                                    "corruptions": str}, keep_default_na=False)


def load_standardized() -> pd.DataFrame:
    """The cleaned records produced by src/standardize.py.

    Kept as CSV rather than parquet on purpose: you can open it, eyeball it,
    and show it to someone without any tooling.
    """
    path = config.DATA_PROCESSED / "standardized.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.standardize` first."
        )
    return pd.read_csv(path, dtype=DTYPE, keep_default_na=False)
