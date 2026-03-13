# ChadGPT

ChadGPT is a hands-on GPT-style language model project built in pure Python and PyTorch. The repository contains a full progression from tokenization and attention mechanics to model implementation, pretraining workflows, and task-specific finetuning.

The code is organized so you can inspect each stage independently while still reusing a shared set of core utilities for model definition, generation, evaluation, and checkpoint loading.

## What is in this repository

- A from-scratch transformer implementation in `chad_gpt.py`, including:
- Causal multi-head attention, transformer blocks, layer normalization, and feed-forward layers.
- Text generation utilities (greedy decoding, temperature, and top-k sampling).
- Training and evaluation helpers for language modeling and classification-style last-token setups.
- Weight-loading utilities for GPT-2 compatible checkpoints.
- GPT-2 download and parameter loading helper in `gpt_download.py`.
- Chapter-based Jupyter notebooks for experimentation across:
- Text preprocessing and tokenization.
- Attention mechanism implementation details.
- GPT model assembly and text generation.
- Pretraining and finetuning workflows.
- Supporting data assets for instruction tuning and spam classification in `chapter-6/` and `chapter-7/`.

## Repository layout

- `chad_gpt.py`: Core model code and utilities.
- `gpt_download.py`: GPT-2 checkpoint download and conversion helpers.
- `chapter-2-*.ipynb` ... `chapter-7-*.ipynb`: Notebook-based experiments and training pipelines.
- `chapter-6/sms_spam_collection/SMSSpamCollection.tsv`: Classification dataset.
- `chapter-7/instruction-data*.json`: Instruction finetuning datasets.
- `the-verdict.txt`: Local text corpus for experiments.

## Quick start

1. Create and activate a virtual environment.
2. Install core dependencies.
3. Open the notebooks or run your own scripts against `chad_gpt.py`.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy tiktoken matplotlib requests tqdm tensorflow
```

## Typical workflow

1. Start in the chapter notebooks to understand each stage of the pipeline.
2. Use `chad_gpt.py` as the reusable module for model architecture and training helpers.
3. Download and load GPT-2 weights when you want to compare behavior against pretrained checkpoints.
4. Experiment with generation parameters (`temperature`, `top_k`, context length) to control output style.

## Notes

- Some assets (for example, large model files) are intentionally not committed by default.
- Training speed and memory usage depend heavily on your local CPU/GPU setup.
- If running on Apple Silicon, verify your PyTorch installation and device configuration before long runs.