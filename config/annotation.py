"""
The human annotation round: rater identities, the controlled vocabularies from
docs/annotation_codebook.md Section 5, and how an annotation batch is composed.

Batch composition is a knob rather than a constant because a uniform sample of D
is roughly 99.8% negative at the configured POSITIVE_PREVALENCE, which leaves an
agreement statistic almost nothing to estimate. See sample.py.
"""

from pathlib import Path

ANNOTATION_DIR = Path("annotation")

RATER_IDS: tuple[str, ...] = ("R1", "R2", "R3")

LABEL_POSITIVE = "LASA-Positive"
LABEL_NEGATIVE = "LASA-Negative"
LABEL_VALUES: tuple[str, ...] = (LABEL_POSITIVE, LABEL_NEGATIVE)

CHANNEL_VALUES: tuple[str, ...] = ("Orthographic", "Phonetic", "Both")

CONFIDENCE_VALUES: tuple[str, ...] = ("High", "Medium", "Low")

COL_PAIR_ID = "pair_id"
COL_ANN_LABEL = "label"
COL_CHANNEL = "channel"
COL_CONFIDENCE = "confidence"
COL_NOTES = "notes"
COL_STRATUM = "stratum"

RATER_FIELDS: tuple[str, ...] = (
    COL_ANN_LABEL,
    COL_CHANNEL,
    COL_CONFIDENCE,
    COL_NOTES,
)

STRATUM_CANDIDATE = "candidate"
STRATUM_NEGATIVE = "negative"
STRATUM_PLACEBO = "placebo"

N_CANDIDATES: int = 120
NEG_PER_POSITIVE: float = 0.5
N_PLACEBO: int = 20

ANNOTATION_SEED: int = 20260718
