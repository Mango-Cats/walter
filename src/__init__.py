"""
walter's source tree, organized in three layers.

    pipeline/   the data transformations, in stage order: clean the registry
                R, sample the unlabeled set U, assemble D, add feature columns.
    proposer/   the LLM-assisted proposal of P. Pipeline-level, but kept apart
                because it is the only stage that talks to a model.
    adapters/   wrappers around external models and binaries (G2P, phoc, tbb,
                the LLM backends). Nothing here knows about drugs or LASA.

The dependency arrow points one way: pipeline and proposer import adapters,
never the reverse. An adapter that needs domain knowledge -- a system prompt,
a column name -- takes it as an argument instead of importing it.

stages.py is the seam that composes the layers into runnable stages, and
artifacts.py resolves a stage's directories to filenames. walter.py owns the
CLI; nothing under src/ parses arguments.
"""
