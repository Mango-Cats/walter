# walter/data

This folder contains datasets used and produced by the Walter pipeline, organized into the following directories:

    - `registries/`: contains cleaned national drug registries and their raw source files:
        - `R_ph.csv`: contains preprocessed Philippine registered human drugs. The
        original dataset is stored in `raw/drug_products.csv` which was saved by
        Erin Chua on December 16, 2026, and updated by Zhean Ganituen on
        October 1, 2026.
        - `R_us.csv`: contains preprocessed United States registered human drugs.
        The original dataset is stored in `raw/Products.txt` which was saved by
        Zhean Ganituen on May 9, 2026.
        - `raw/`: raw registry dumps before preprocessing:
            - `drug_products.csv`: Philippine FDA registered drug products dataset.
            - `Products.txt`: US FDA registered drug products dataset.

    - `ococosda2026/`: contains confusable drug name datasets constructed for the
    OCOCOSDA 2026 study, seeded from ISMP's List of Confused Drug Names (`P_us.csv`)
    with an equal-count, phoc-vetted hard-negative set:
        - `D.csv`: assembled confusable and hard-negative pairs with English and
        Filipino IPA transcriptions (`x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label`).
        - `D_pho.csv`: `D.csv` augmented with phonetic and orthographic similarity
        feature columns computed via `phoc`.

    - `old/`: archived legacy datasets kept for historical reference:
        - `P_ph.csv`: legacy predefined Philippine confusable drug pairs.
