"""
The LLM-assisted proposal of P.

    prompt      the system prompt and the user-turn constructor
    inference   fuzzy-matches candidates out of the registry, asks a backend
                which are genuine confusibles, and writes the JSON P

The proposer augments rather than invents: it is seeded with predefined LASA
pairs and the seed pair always survives into the output. The model chooses
among registry names it is shown; it never supplies a name of its own.

Which backend runs is config.USE_API_MODEL's call -- see src/adapters/llm.
"""
