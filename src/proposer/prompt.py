"""
The system prompt and a function for constructing the prompt (putting)
pieces together).
"""

SYSTEM_PROMPT = """You are a pharmacist and an expert in Filipino phonology.

Task:
Identify drug names that are LOOK-ALIKE or SOUND-ALIKE (LASA) relative to a target drug.

Definitions:
- LOOK-ALIKE: Orthographically similar (e.g., similar spelling, shared prefixes/suffixes, edit distance, visual confusion risk)
- SOUND-ALIKE: Phonetically similar when spoken in English or Filipino (consider syllable structure, stress, consonant/vowel substitution common in Filipino speech)

Constraints:
- Only select drug names that appear in the provided dataset
- Do NOT invent or modify drug names
- Prioritize high-risk confusion pairs (clinically plausible misread/misheard cases)
- Avoid duplicates
- Avoid the exact same drug as the input

Selection Criteria (rank implicitly):
1. Phonetic similarity (Filipino + English pronunciation)
2. Orthographic similarity
3. Risk of real-world confusion in prescribing/dispensing

Output Format:
- Return EXACTLY N drug names
- One per line
- No numbering
- No explanations
- No extra text, punctuation, or quotes

If fewer than N valid candidates exist, return only the valid ones.
"""


def construct_user_prompt(
    drug_name: str,
    dataset: str,
    n: int=1,
) -> str:
    """
    This constructs a user prompt given a specific `drug_name`, a
    set of candidates from `dataset`, and the number `n` of drugs to
    select from the `dataset`.
    """
    # FIXME(zhean): we should have some few-shot examples here i think
    user_prompt = f"""
Target Drug:
{drug_name}

Dataset (candidate drugs):
{dataset}

Task:
From the dataset above, return EXACTLY {n} drug names that are most likely to be confused with the target drug based on LOOK-ALIKE or SOUND-ALIKE properties.

Reminder:
- Only choose from the dataset
- One drug per line
- No explanations
"""

    return user_prompt
