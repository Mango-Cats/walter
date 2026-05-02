"""
The system prompt and a function for constructing the prompt (putting)
pieces together).
"""

SYSTEM_PROMPT = """You are a pharmacist and an expert in Filipino phonology.

Task:
Identify drug names that are LOOK-ALIKE or SOUND-ALIKE (LASA).

STRICT OUTPUT FORMAT (MANDATORY):
- Output ONLY the drug names
- One per line
- NO explanations
- NO reasoning
- NO sentences
- NO extra text

Rules:
- Only choose from the provided dataset
- Do NOT modify names
- Do NOT repeat the input drug
- Prefer high-risk confusion pairs
"""


def construct_user_prompt(drug_name: str, dataset: str, n: int = 1) -> str:
    """
    This constructs a user prompt given a specific `drug_name`, a
    set of candidates from `dataset`, and the number `n` of drugs to
    select from the `dataset`.
    """
    return f"""Target Drug:
{drug_name}

Candidate Drugs:
{dataset}

Task:
Return EXACTLY {n} drug names from the dataset that are most likely to be confused with the target drug.

Output:
"""
