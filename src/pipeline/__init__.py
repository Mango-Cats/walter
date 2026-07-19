"""
The data transformations, in the order the pipeline runs them.

    preprocessing   R  -- load and clean the drug registry
    noise           U  -- two-tier similarity-filtered sampling of unlabeled
                          pairs, constrained per cluster
    clustering         -- union-find backing noise's cluster bookkeeping
    dataset         D  -- merge P and U, transcribe, deduplicate
    features           -- append the orthographic META_FEATURES onto D

Every module here takes and returns DataFrames. Reading and writing the
canonical artifact files is stages.py's job, not theirs.
"""
