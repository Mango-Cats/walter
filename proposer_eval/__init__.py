"""
proposer_eval: quiz the proposer's candidate gate against the annotated gold pairs.

Question it answers: if `extract_candidates` (src/proposer/inference.py) used
pho's Soundex and/or the TagaBaybay nativizer in place of the RapidFuzz WRatio
gate, would it find the same confusibles, more of them, or fewer?

Run from the repo root:  uv run python -m proposer_eval
"""
