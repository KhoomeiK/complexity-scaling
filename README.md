<p align="center">
  <img src="https://raw.githubusercontent.com/khoomeik/complexity-scaling/main/.github/scaling_contours.png" height="300" alt="Comparison of parameter-data scaling contours for datasets of 2 different gzip-compressibilities" />
</p>
<!-- <h2 align="center">
  gzip Predicts Data-dependent Scaling Laws
</h2> -->
<p align="center">
<a href="https://x.com/khoomeik/status/1795477359933706272">🐦 Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://arxiv.org/abs/2405.16684">📄 Arxiv</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://huggingface.co/khoomeik">🤗 Datasets</a>
</p>
<p align="center">
<a href="https://reworkd.ai/">🔗 Multimodal CodeGen for Web Data Extraction</a>
</p>

# `gzip` Predicts Data-dependent Scaling Laws

This is the official code for *`gzip` Predicts Data-dependent Scaling Laws* (under review at NeurIPS 2024).

We find that:
1. scaling laws are sensitive to differences in data complexity
2. `gzip`, a compression algorithm, is an effective predictor of how data complexity impacts scaling properties

Our data-dependent scaling law's compute-optimal frontier increases in dataset size preference (over parameter count preference) as training data becomes more complex (harder to compress).

## Code Overview
- `data_gen.py`: create PCFGs with specified syntactic properties and sample text datasets from them
- `data_utils.py`: `gzip`-compressibility measurement, tokenization & HuggingFace tooling, dataloaders, etc.
- `training.py`: run a single training run given model and dataset, returning loss at each train step
- `main.py`: run a set of training runs across datasets & model sizes (hackily GPU-parallelized with threading)
- `fsdp_training.py`: for running bigger jobs with cleaner data loading & FSDP training

Upon request via email, we can also provide:
- JSONL records of all training runs (this is large and can't fit on GitHub)
- the Jupyter Notebook used to fit scaling laws from training runs and generate all visuals
