# walter/scripts

The following is a sample of how to use the prompt evaluation pipeline.

Outputs are organized under `results/eval/` by stage: `pool/`, `questionnaire/`,
`proposer/`, `report/`. `results/eval/hard_cases.txt` (a curated `--targets`
list) sits at the top level since it's a hand-written input, not a stage output.

1. Sample a pool of drugs to use as "target drugs"

uv run python scripts/eval_sample.py \
    --gold data/P_us.csv \
    --targets results/eval/hard_cases.txt \
    --registry data/R_us.csv --n-easy 200 \
    --output results/eval/pool/pool_targeted.json

2. Create the questionnaire, mimicing the format of the actual data construction.

uv run python scripts/eval_candidates.py \
    --pool results/eval/pool/pool_targeted.json \
    --n-fuzzy 5 --n-random 10 --n-easy 5 --seed 0 \
    --output results/eval/questionnaire/questionnaire_targeted.json

3. Run the proposal using the baseline/default prompt using model "deepseek-v4-pro".

uv run python scripts/eval_propose.py \
    --input results/eval/questionnaire/questionnaire_targeted.json \
    --system-prompt prompts/baseline.txt \
    --model deepseek-v4-pro \
    --output results/eval/proposer/proposer_targeted_baseline.json \
    --concurrency 26

4. Run the proposal using a modified baseline prompt using model "deepseek-v4-pro".

uv run python scripts/eval_propose.py \
    --input results/eval/questionnaire/questionnaire_targeted.json \
    --system-prompt prompts/baseline_mod.txt \
    --model deepseek-v4-pro \
    --output results/eval/proposer/proposer_targeted_baseline_mod.json \
    --concurrency 26

5. Evaluate baseline results.

uv run python scripts/eval_score.py \
    --input results/eval/proposer/proposer_targeted_baseline.json \
    --tiers-pool results/eval/pool/pool_targeted.json \
    --output results/eval/report/report_targeted_baseline.json

6. Evaluate modified baseline results.

uv run python scripts/eval_score.py \
    --input results/eval/proposer/proposer_targeted_baseline_mod.json \
    --tiers-pool results/eval/pool/pool_targeted.json \
    --output results/eval/report/report_targeted_baseline_mod.json
