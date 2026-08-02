"""
The clerical review queue.

Every pair the model refused to decide lands here, one at a time, side by
side, with the fields that disagree marked. A reviewer marks each pair as the
same person or two different people, and that decision is saved as a new
labelled example.

This is what closes the loop. The labels collected here are exactly the
training data the model was short of, namely hard and ambiguous cases rather
than easy synthetic ones, so retraining on them improves the next run.

Run:  python -m streamlit run app.py

Invoked through `python -m` rather than the bare `streamlit` command, because
pip installs console scripts to a directory that is not on PATH under the
Microsoft Store build of Python, and `streamlit run` fails there with a
CommandNotFoundException. Going through the interpreter always works.
"""

import html

import pandas as pd
import streamlit as st

import config
from src.load import load_standardized

LABELS_PATH = config.DATA_PROCESSED / "human_labels.csv"

FIELDS = [
    ("Source system", "source_system"),
    ("First name", "first_name"),
    ("Middle name", "middle_name"),
    ("Last name", "last_name"),
    ("Date of birth", "dob"),
    ("Email", "email"),
    ("Phone", "phone"),
    ("Address", "address_line"),
    ("City", "city"),
    ("Pincode", "pincode"),
    ("Record created", "created_date"),
]

st.set_page_config(
    page_title="Single Customer View",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLES = """
<style>
  .block-container { padding-top: 2.5rem; max-width: 1100px; }

  .gr-header {
      border-bottom: 1px solid rgba(128, 128, 128, 0.25);
      padding-bottom: 0.9rem;
      margin-bottom: 1.6rem;
  }
  .gr-title {
      font-size: 1.45rem;
      font-weight: 600;
      letter-spacing: -0.01em;
      margin: 0;
  }
  .gr-subtitle {
      font-size: 0.875rem;
      color: rgba(128, 128, 128, 0.95);
      margin-top: 0.35rem;
  }

  table.gr-compare {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      margin-top: 0.5rem;
  }
  table.gr-compare th {
      text-align: left;
      font-weight: 600;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: rgba(128, 128, 128, 0.95);
      padding: 0.5rem 0.85rem;
      border-bottom: 1px solid rgba(128, 128, 128, 0.3);
  }
  table.gr-compare td {
      padding: 0.55rem 0.85rem;
      border-bottom: 1px solid rgba(128, 128, 128, 0.15);
      vertical-align: top;
  }
  table.gr-compare td.gr-label {
      width: 20%;
      color: rgba(128, 128, 128, 0.95);
  }
  table.gr-compare tr.gr-differs td.gr-label { color: inherit; font-weight: 600; }
  table.gr-compare tr.gr-differs td.gr-value {
      background: rgba(191, 138, 0, 0.13);
      font-weight: 500;
  }
  table.gr-compare td.gr-value { width: 40%; }
  .gr-missing { color: rgba(128, 128, 128, 0.7); font-style: italic; }
  .gr-record-id {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.78rem;
      font-weight: 400;
      color: rgba(128, 128, 128, 0.95);
      display: block;
      margin-top: 0.15rem;
  }

  .gr-note {
      font-size: 0.82rem;
      color: rgba(128, 128, 128, 0.95);
      line-height: 1.5;
  }
  .gr-verdict {
      border: 1px solid rgba(128, 128, 128, 0.3);
      border-left: 3px solid rgba(128, 128, 128, 0.55);
      padding: 0.7rem 0.95rem;
      font-size: 0.86rem;
      line-height: 1.5;
  }
  .gr-verdict strong { font-weight: 600; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


@st.cache_data
def load_queue():
    decisions = pd.read_csv(config.DATA_PROCESSED / "decisions.csv")
    records = load_standardized()
    queue = decisions[decisions.band == "review"].sort_values("score", ascending=False)
    return queue.reset_index(drop=True), records, len(decisions)


@st.cache_data
def load_llm_verdicts():
    path = config.DATA_PROCESSED / "llm_verdicts.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).set_index(["record_id_a", "record_id_b"])


def existing_labels() -> pd.DataFrame:
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=["record_id_a", "record_id_b", "human_label", "score"])


def save_label(row, label: int) -> None:
    new = pd.DataFrame([{
        "record_id_a": row.record_id_a,
        "record_id_b": row.record_id_b,
        "human_label": label,
        "score": row.score,
    }])
    combined = pd.concat([existing_labels(), new], ignore_index=True)
    combined.drop_duplicates(
        subset=["record_id_a", "record_id_b"], keep="last"
    ).to_csv(LABELS_PATH, index=False)


def render_cell(value: str) -> str:
    if not value:
        return '<span class="gr-missing">not provided</span>'
    return html.escape(value)


def comparison_table(record_a, record_b, id_a: str, id_b: str) -> str:
    rows = []
    for label, field in FIELDS:
        value_a = str(record_a.get(field, "") or "").strip()
        value_b = str(record_b.get(field, "") or "").strip()
        differs = value_a.lower() != value_b.lower()
        css = ' class="gr-differs"' if differs else ""
        rows.append(
            f"<tr{css}>"
            f'<td class="gr-label">{html.escape(label)}</td>'
            f'<td class="gr-value">{render_cell(value_a)}</td>'
            f'<td class="gr-value">{render_cell(value_b)}</td>'
            f"</tr>"
        )
    return (
        '<table class="gr-compare">'
        "<thead><tr>"
        "<th>Field</th>"
        f'<th>Record A<span class="gr-record-id">{html.escape(id_a)}</span></th>'
        f'<th>Record B<span class="gr-record-id">{html.escape(id_b)}</span></th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


queue, records, total_decisions = load_queue()
llm_verdicts = load_llm_verdicts()
by_record = records.set_index("record_id")

automated = 1 - len(queue) / total_decisions

st.markdown(
    '<div class="gr-header">'
    '<p class="gr-title">Clerical review queue</p>'
    f'<p class="gr-subtitle">The matcher resolved {automated:.1%} of '
    f"{total_decisions:,} candidate pairs without assistance. "
    f"The {len(queue):,} pairs below scored between {config.LOWER_THRESHOLD} and "
    f"{config.UPPER_THRESHOLD} and were escalated for a decision.</p>"
    "</div>",
    unsafe_allow_html=True,
)

if "index" not in st.session_state:
    st.session_state.index = 0

labels = existing_labels()
reviewed = len(labels)

with st.sidebar:
    st.markdown("**Progress**")
    st.metric("Queue length", f"{len(queue):,}")
    st.metric("Reviewed", f"{reviewed:,}")
    if reviewed:
        st.metric("Marked same person", f"{int(labels.human_label.sum()):,}")
    st.progress(min(reviewed / max(len(queue), 1), 1.0))

    st.divider()
    jump = st.number_input(
        "Go to pair", min_value=0, max_value=max(len(queue) - 1, 0),
        value=st.session_state.index, step=1,
    )
    if jump != st.session_state.index:
        st.session_state.index = int(jump)

    st.divider()
    st.markdown(
        '<p class="gr-note">Each decision is written to human_labels.csv and '
        "becomes training data for the next run. Ambiguous cases carry far more "
        "information than easy ones.</p>",
        unsafe_allow_html=True,
    )

if st.session_state.index >= len(queue):
    st.success("Queue complete. Retrain to fold these labels back into the model.")
    st.stop()

row = queue.iloc[st.session_state.index]
record_a = by_record.loc[row.record_id_a]
record_b = by_record.loc[row.record_id_b]

left, right = st.columns([1, 2.4], gap="large")

with left:
    st.metric(f"Pair {st.session_state.index + 1:,} of {len(queue):,}",
              f"{row.score:.3f}", help="Model confidence that these are the same person")
    st.markdown(
        '<p class="gr-note">The model is measurably unreliable in this score '
        "range, which is the reason the pair was escalated rather than decided. "
        "Treat the score as one input, not a verdict.</p>",
        unsafe_allow_html=True,
    )

with right:
    if llm_verdicts is not None and (row.record_id_a, row.record_id_b) in llm_verdicts.index:
        verdict = llm_verdicts.loc[(row.record_id_a, row.record_id_b)]
        call = "Same person" if verdict.llm_match else "Different people"
        st.markdown(
            f'<div class="gr-verdict"><strong>LLM adjudicator: {call}</strong> '
            f"(confidence {verdict.llm_confidence})<br>"
            f"{html.escape(str(verdict.llm_reason))}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="gr-verdict">No LLM verdict for this pair. '
            "Run <code>python -m src.llm_adjudicator</code> to adjudicate the "
            "review band.</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    comparison_table(record_a, record_b, row.record_id_a, row.record_id_b),
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="gr-note" style="margin-top:0.6rem">Highlighted rows are fields '
    "where the two records disagree.</p>",
    unsafe_allow_html=True,
)

st.write("")
actions = st.columns([1, 1, 1, 3])
if actions[0].button("Same person", use_container_width=True, type="primary"):
    save_label(row, 1)
    st.session_state.index += 1
    st.rerun()
if actions[1].button("Different people", use_container_width=True):
    save_label(row, 0)
    st.session_state.index += 1
    st.rerun()
if actions[2].button("Skip", use_container_width=True):
    st.session_state.index += 1
    st.rerun()
