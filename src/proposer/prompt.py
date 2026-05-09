"""
System prompt and user prompt constructor for the local LLM LASA proposer.
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


def construct_user_prompt(drug_name: str, candidates: str, n: int = 1) -> str:
    """
    Build the user-turn prompt for the LLM.

    Args:
        drug_name:  The target drug name.
        candidates: Newline-separated candidate drug names.
        n:          Number of confusibles to request.
    """
    return (
        f"Target Drug:\n{drug_name}\n\n"
        f"Candidate Drugs:\n{candidates}\n\n"
        f"Task:\nReturn EXACTLY {n} drug names from the dataset "
        f"that are most likely to be confused with the target drug.\n\n"
        f"Output:\n"
    )
