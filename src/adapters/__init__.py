"""
Wrappers around external models and binaries.

    g2p/    grapheme-to-phoneme transcription (English via phonemizer,
            Filipino via a Phonetisaurus FST)
    llm/    the proposer's inference backends -- a local transformers model
            and the DeepSeek API
    phoc    the phoc binary in bin/, for phonetic-similarity features
    tbb     the tbb-cli binary, for Filipino nativization

These modules know how to drive a tool, not what walter is for. Domain
knowledge (system prompts, which languages to transcribe) arrives as
arguments so the dependency arrow keeps pointing away from pipeline/.
"""
