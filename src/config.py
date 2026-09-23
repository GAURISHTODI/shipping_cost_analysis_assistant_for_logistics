"""Central, documented constants for the cost-anomaly pipeline.

Every tunable knob lives here so a reviewer can see, in one place,
every judgment call the pipeline makes.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

SHIPMENTS_CSV = DATA_DIR / "shipment_records.csv"
NOTES_CSV = DATA_DIR / "context_notes.csv"
SAMPLE_FORMAT_CSV = DATA_DIR / "sample_output_format_v2.csv"
OUTPUT_CSV = OUTPUT_DIR / "output.csv"

# --- Weekly aggregation --------------------------------------------------
# Weeks run Monday -> Sunday. week_of is the Monday date.
WEEK_ANCHOR = "W-MON"  # pandas resample anchor meaning "week ending Monday"

# --- Baselines -------------------------------------------------------------
ROLLING_WINDOW_WEEKS = 8  # trailing own-history window, strictly prior weeks

# --- Anomaly flagging -------------------------------------------------------
# A route-week is a "rising & anomalous" CANDIDATE if it is at least this much
# above ITS OWN 8-week trailing average, OR at least this much above the
# same-week peer average for its route_type. Chosen (and verified) against
# the three worked examples in sample_output_format_v2.csv, where the
# smallest triggering deviation was +21% (peer) / -- the largest non-triggering
# comparisons in the underlying data sit well below 15%, so 15% is a clean,
# documented separator rather than a value tuned to hit the samples exactly.
ANOMALY_THRESHOLD_PCT = 0.15

# --- Note-grounding ----------------------------------------------------
# When a note gives no explicit "<date> to <date>" range in its text, its
# effect is assumed to last exactly the ISO week (Mon-Sun) containing the
# note's own `date` field -- NOT indefinitely, even if the note says an
# effect is "starting this week". Guessing a longer, open-ended window risks
# justifying a cost rise months later purely because *some* unrelated note
# once mentioned a price rise -- exactly the fabrication failure mode the
# brief penalizes. See README "Note grounding" section for the worked
# example (N003 diesel-price note) that forced this choice.
DEFAULT_NOTE_WINDOW_DAYS = 6  # Monday..Sunday inclusive == 6 days after start

# Phrases that mean "this note does NOT justify a cost increase", even if it
# is topically about the right route/date. Checked case-insensitively.
# Matching any one of these forces polarity = negative, overriding any
# cost-positive language elsewhere in the same note.
NEGATIVE_PATTERNS = [
    r"not (?:be )?significantly affected",
    r"\bnot affected\b",
    r"no significant disruptions?",
    r"no major disruptions?",
    r"without a rate change",
    r"remained normal",
    r"remained stable",
    r"returned to normal",
    r"improved road conditions",
    r"improved conditions",
    r"not part of (?:this|the) dataset",
    r"costs? were not",
]

# Phrases that indicate the note describes a genuine cost-increasing event.
# Only checked if no NEGATIVE_PATTERNS matched.
POSITIVE_PATTERNS = [
    r"surcharge",
    r"\brose\b",
    r"rising",
    r"push(?:ed|ing)? up",
    r"higher (?:trip )?costs?",
    r"detours?",
    r"disrupt(?:ed|ion)?",
    r"costlier",
    r"pricier",
    r"price(?:s)? (?:rose|increased|went up)",
]

RETRIEVAL_TOP_K = 3
