# ReWiSe: Relation-Wise Self-consistency for LLM Probing

**Authors:** Edouard Albert-Roulhac and Amal Zouaq  
**Affiliation:** LAMA-WeST Lab, Polytechnique Montréal  
**Contact:** <edouard.albert-roulhac@polymtl.ca>

## Participation to LM-KBC Challenge @ ISWC 2025

**Competition URL:** <https://lm-kbc.github.io/challenge2025/>

This repository contains our submission implementing Relation-Wise Self-consistency with synthetic Chain-of-Thought (CoT) generation for knowledge base construction.

## Quick Start

### Dependencies

```bash
pip install -r requirements.txt
```

### Generate Synthetic CoT Data (can be skipped if using provided SyntheticCoT)

```bash
# Generate synthetic CoT using Lambda API (can also be done locally if sufficient hardware)
python src/synthetic_cot_lambda.py \
    --dataset dataset2025 \
    --split train \
    --mode nohelp \
    --few-shot 2 \
    --temperature 1.0 \
    --output data/dataset2025/cot/nohelp_synthetic_cot/raw/synthetic_cot.csv

# Post-process and clean the synthetic CoT data
python src/process_synthetic_cot.py \
    --input data/dataset2025/cot/nohelp_synthetic_cot/raw/synthetic_lambda.csv \
    --dataset dataset2025 \
    --mode nohelp \
    --quality-filter correct,incomplete
```

### Run Inference with CoT and Self-Consistency

```bash
# Submit inference job with Qwen3 8B and 20-way self-consistency
sbatch scripts/submit_inference.sh \
    --model q_8i \
    --dataset dataset2025 \
    --split test \
    --cot \
    --cot-source synthetic_lambda \
    --n-consistency 20
```

### Create Competition Submission

```bash
# Convert predictions to competition format
python src/create_submission.py [path_to_predictions]
```
