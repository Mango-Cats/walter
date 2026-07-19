"""
The human annotation round: rater identities, the controlled vocabularies from
docs/annotation_codebook.md Section 5, and how an annotation batch is composed.

Batch composition is a knob rather than a constant because a uniform sample of D
is roughly 99.8% negative at the configured POSITIVE_PREVALENCE, which leaves an
agreement statistic almost nothing to estimate. See sample.py.
"""

from pathlib import Path

# Rater output is produced by people and cannot be regenerated, so this lives
# outside RESULTS_DIR and is tracked.
ANNOTATION_DIR = Path("annotation")

RATER_IDS: tuple[str, ...] = ("R1", "R2", "R3")

# Section 4. Binary to match the downstream classifier target; an "uncertain"
# value is deliberately absent, and uncertainty goes in confidence/notes.
LABEL_POSITIVE = "LASA-Positive"
LABEL_NEGATIVE = "LASA-Negative"
LABEL_VALUES: tuple[str, ...] = (LABEL_POSITIVE, LABEL_NEGATIVE)

# Section 5. `channel` records which similarity channel drove the decision, so
# disagreement can later be traced to the phonetic channel specifically.
CHANNEL_VALUES: tuple[str, ...] = ("Orthographic", "Phonetic", "Both")

CONFIDENCE_VALUES: tuple[str, ...] = ("High", "Medium", "Low")

# Sheet columns. COL_ANN_LABEL collides by name with schema.COL_LABEL, but the
# two never share a frame: the blinded sheet excludes D's 0/1 label entirely,
# and here the value is one of LABEL_VALUES.
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

# Batch strata. Placebo pairs are unrelated names that no reader should mark
# positive, so a rater agreeing with the others on them is evidence of attention
# rather than straight-lining (Section 8.1).
STRATUM_CANDIDATE = "candidate"
STRATUM_NEGATIVE = "negative"
STRATUM_PLACEBO = "placebo"

N_CANDIDATES: int = 120
NEG_PER_POSITIVE: float = 0.5
N_PLACEBO: int = 20

# Separate from sampling.SEED so re-drawing a batch never perturbs how U is built.
ANNOTATION_SEED: int = 20260718
