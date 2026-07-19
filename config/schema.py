"""
Column names and label values: the shape of every dataset the pipeline
passes between stages. Nothing here is a tunable knob; changing a name here
renames a column everywhere.
"""

# Single-column name in the raw / cleaned registry
REGISTRY_COL: str = "drug_name"

# Pair dataset columns
COL_X1: str = "x_1"  # drug name A
COL_X2: str = "x_2"  # drug name B

# phoc's wire format ONLY. The phoc binary hardcodes t_1/t_2 as the names of
# the transcription columns it reads, so these never appear in a dataset we
# write -- src/adapters/phoc.py materializes them into a temp CSV, one run per
# language. The real, persisted transcription columns are the per-language
# ones below.
COL_T1: str = "t_1"
COL_T2: str = "t_2"

# Persisted per-language IPA transcriptions of x_1 / x_2.
COL_T_ENG_1: str = "t_eng_1"  # English (en-us), src/adapters/g2p/eng.py
COL_T_ENG_2: str = "t_eng_2"
COL_T_FIL_1: str = "t_fil_1"  # Filipino (Tagalog), src/adapters/g2p/fil.py
COL_T_FIL_2: str = "t_fil_2"

# The languages every transcription-dependent feature is computed once per.
# Adding a language here fans out the aline columns automatically (src/adapters/phoc.py
# emits <config_stem>_<lang> for each) -- no other file needs to change.
TRANSCRIPTION_LANGS: dict[str, tuple[str, str]] = {
    "eng": (COL_T_ENG_1, COL_T_ENG_2),
    "fil": (COL_T_FIL_1, COL_T_FIL_2),
}

COL_LABEL: str = "label"  # 1 = known positive (LASA), 0 = unlabeled

# P.csv (input) must have exactly these two columns
P_INPUT_COLS: list[str] = [COL_X1, COL_X2]

# Labels
POSITIVE_LABEL: int = 1
UNLABELED_LABEL: int = 0
