# Differentiated Token-Masking for Self-Curated Difficulty Alignment

Master's thesis framework investigating **self-curated dataset optimization** through the lens of token-masking mechanics. Compares two distinct scoring strategies — **Input Masking (D_in)** and **Output Masking (D_out)** — as difficulty metrics for selective SFT data curation on lightweight language models.

**Baseline model:** [Qwen-2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (0.49B params)

## Overview

The pipeline operates in three phases:

| Phase | Description | Data Source |
|-------|-------------|-------------|
| **Curation** | Score 20k samples via masking strategies, select median-difficulty IQR subset (5k) | Alpaca-Cleaned (semantic), GSM8K (logical) |
| **Training** | QLoRA fine-tuning on RTX 3090 / single-GPU hardware | Curated 5k JSONL files |
| **Evaluation** | Zero-shot benchmarks + Backward Transfer (BWT) perplexity | ARC-C, HellaSwag, MMLU, GSM8K, MATH |

### Model Variants

1. **Baseline** — Zero-shot Qwen-2.5-0.5B (no tuning)
2. **Control** — Fine-tuned on random 5k samples (no curation)
3. **Input-Masked** — Curated via Strategy A (D_in): masks high-information prompt tokens
4. **Output-Masked** — Curated via Strategy B (D_out): masks sequential target blocks

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ GPU (tested on RTX 3090, targets ~8 GB VRAM minimum)
- [Git LFS](https://git-lfs.com/) (optional, for model weights)

### Quick Install

```bash
chmod +x install_deps.sh
./install_deps.sh
```

### Manual Install

```bash
# 1. PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Pipeline dependencies
pip install -r requirements.txt

# 3. SpaCy English model
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Full pipeline (all phases sequentially)
python main.py --curate --train --evaluate

# Individual phases
python main.py --curate      # Phase 1: score + curate datasets
python main.py --train       # Phase 2: QLoRA fine-tuning
python main.py --evaluate    # Phase 3: benchmarks + BWT
```

### Outputs

| Path | Contents |
|------|----------|
| `curated_5k_*.json` | Curated training subsets (5k samples each) |
| `scores_*.jsonl` | Raw difficulty scores with resume support |
| `./results/*` | Training checkpoints (auto-resume on interrupt) |
| `./final_models/*` | Final LoRA adapter weights |
| `final_evaluation_matrix.json` | Benchmark results + BWT scores |
| `curation_errors.log` | Skipped-sample log from curation phase |

## Project Structure

```
├── main.py                  # CLI orchestrator (--curate, --train, --evaluate)
├── data_loader.py           # Loads Alpaca-Cleaned, GSM8K, and Wikitext holdout
├── masking_metrics.py       # D_in and D_out scoring functions
├── curation.py              # IQR-based difficulty filtering + control subset generator
├── train_qlora.py           # 4-bit NF4 QLoRA with Flash Attention 2
├── evaluate.py              # vLLM benchmarks + BWT perplexity on Wikitext
├── requirements.txt         # Python dependencies
├── install_deps.sh          # One-shot dependency installer
└── docs/                    # Methodology document
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | Qwen-2.5-0.5B |
| Quantization | 4-bit NF4 (double quant) |
| LoRA rank / alpha | 16 / 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2e-5 |
| Batch size | 4 × 4 accumulation (effective 16) |
| Max sequence length | 1024 |
| Epochs | 4 |
| Scheduler | Cosine annealing |
| Curation subset | 5,000 (IQR-filtered from 20,000) |

## Benchmarks

### Semantic Track

- **ARC-Challenge** — Science reasoning
- **HellaSwag** — Commonsense NL inference
- **MMLU** — Multitask language understanding

### Logical Track

- **GSM8K** — Multi-step arithmetic reasoning
- **MATH** — Advanced mathematical derivation

### Forgetting Metric

Backward Transfer (BWT) quantified via perplexity delta on Wikitext-2 holdout set against the zero-shot baseline.

## License

This project is part of an academic Master's thesis. See `docs/Methodology.txt` for the full methodology document.
