"""Standard column names and dataset labels used across the Walter pipeline.

This module defines the table structure for drug pairs, pronunciation columns,
and numeric classification labels:
    - 1  : Confusable (positive LASA pair)
    - 0  : Non-confusable (unlabeled sample)
    - -1 : Explicitly rejected pair (soft label)
"""

REGISTRY_COL: str = "drug_name"

COL_X1: str = "x_1"
COL_X2: str = "x_2"

COL_T1: str = "t_1"
COL_T2: str = "t_2"

COL_T_ENG_1: str = "t_eng_1"
COL_T_ENG_2: str = "t_eng_2"
COL_T_FIL_1: str = "t_fil_1"
COL_T_FIL_2: str = "t_fil_2"

TRANSCRIPTION_LANGS: dict[str, tuple[str, str]] = {
    "eng": (COL_T_ENG_1, COL_T_ENG_2),
    "fil": (COL_T_FIL_1, COL_T_FIL_2),
}

COL_LABEL: str = "label"

P_INPUT_COLS: list[str] = [COL_X1, COL_X2]

POSITIVE_LABEL: int = 1
UNLABELED_LABEL: int = 0
NEGATIVE_LABEL: int = -1
