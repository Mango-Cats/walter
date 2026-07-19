"""
Grapheme-to-phoneme transcription.

    eng         English IPA via phonemizer/espeak
    fil         Filipino IPA via a Phonetisaurus FST (bin/cwik_model.fst)
    client      one transcribe_dataframe() signature over either language
    transcribe  transcribes both names of every pair, for the languages in
                config.TRANSCRIPTION_LANGS

Callers should go through transcribe or client rather than reaching for a
language module directly; the two languages have different backends but the
same interface, and that is the point of the split.
"""
