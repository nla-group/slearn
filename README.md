# slearn: learning symbolic sequences

[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![Codecov Workflow](https://github.com/nla-group/slearn/actions/workflows/unittests.yml/badge.svg)](https://github.com/nla-group/slearn/actions/workflows/unittests.yml)
[![PyPI Version](https://badge.fury.io/py/slearn.svg)](https://badge.fury.io/py/slearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conda Version](https://anaconda.org/conda-forge/slearn/badges/version.svg)](https://anaconda.org/conda-forge/slearn)
[![Documentation Status](https://readthedocs.org/projects/slearn/badge/?version=latest)](https://slearn.readthedocs.io/en/latest/?badge=latest)

`slearn` is a research package for symbolic sequence generation, symbolic time-series representation, string-distance evaluation, and controlled sequence-learning experiments. It was originally developed around LZW-controlled symbolic strings and LSTM/GRU forecasting; the `exps/` directory now includes a unified benchmark for recurrent, Transformer, efficient-attention, and RWKV-style models on the same next-token task.

## Install

For the core package:

```bash
pip install slearn
# or
conda install -c conda-forge slearn
```

For the manuscript experiments, use an isolated environment because modern sequence models may require a newer PyTorch build:

```bash
git clone https://github.com/chenxinye/slearn.git
cd slearn
bash scripts/install_experiment_deps.sh
source .venv/bin/activate
```

Set `INSTALL_RWKV_TRAINER=1` before running the script if you also want the separate `rwkv-trainer` package. The default RWKV baseline in `exps/models.py` is a compact PyTorch RWKV-style time-mixing block so that it can run inside the same supervised batch loop as the other models.

## Core Features

- LZW-controlled symbolic string generation through `lzw_string_generator` and `lzw_string_seeds`.
- String distances and similarities, including Damerau-Levenshtein, Jaro-Winkler, Hamming, cosine, LCS, Dice, and Smith-Waterman variants.
- Symbolic time-series transforms, including SAX, SAX-TD, eSAX, mSAX, aSAX, and ABBA-style representations.
- A unified experiment harness for finite-context next-symbol prediction and recursive symbolic rollout.

## LZW String Generation

```python
from slearn import lzw_string_generator, lzw_string_seeds

seed, complexity = lzw_string_generator(
    nr_symbols=4,
    target_complexity=30,
    priorise_complexity=True,
    random_state=2,
)
print(seed, complexity)

library = lzw_string_seeds(
    symbols=[2, 4, 6, 8],
    complexity=[10, 30, 50],
    iterations=3,
    random_state=42,
)
print(library.head())
```

The generator first ensures that the requested alphabet appears in the seed and then appends symbols until the LZW complexity of the reduced string reaches the target. These seeds can be periodically repeated to create controlled symbolic sequences with known alphabet size, target complexity, and sequence length.

## Symbolic Benchmark

The main experiment script is:

```bash
python exps/symbolic_sequence_benchmark.py
```

The script trains a model on windows from the observed prefix of a repeated LZW seed. For a sequence `x_1, ..., x_N`, context length `w`, and forecast horizon `h`, it trains on pairs `(x_{t-w+1:t}, x_{t+1})` from `x_1, ..., x_{N-h}` and evaluates:

- held-out next-token loss and accuracy on the prefix;
- recursive rollout from `x_{N-h-w+1:N-h}` for `h` steps;
- normalized Damerau-Levenshtein distance (`DL`, lower is better);
- normalized Jaro-Winkler distance (`JW`, lower is better);
- parameter count, training time, epoch count, and peak GPU memory.

Supported model names:

```text
LSTM GRU minGRU minLSTM Transformer BERT GPT LinearAttention Performer RWKV
```

The newer baselines are implemented as:

- `minGRU`: `minGRU-pytorch` wrapped as a sequence classifier.
- `minLSTM`: a compact PyTorch implementation of the minimal LSTM gating equations.
- `LinearAttention`: `linear-attention-transformer` with causal linear attention.
- `Performer`: `performer-pytorch` with causal FAVOR+ attention.
- `RWKV`: a lightweight RWKV-style time-mixing block for from-scratch symbolic experiments.

### Quick Check

```bash
python exps/symbolic_sequence_benchmark.py --smoke
```

This runs a tiny configuration to verify imports, data preparation, training, and rollout.

### Default Workload

The default benchmark is intentionally a pilot-sized run:

```text
8 models x 4 alphabet sizes x 5 complexities x 2 seeds x 2 runs = 640 model fits
```

Each fit uses one model width, one learning rate, one weight decay, one sequence length, and at most 200 epochs with patience-based early stopping. This is still substantial, but it is small enough to shard across a Slurm array. A larger sweep with 10 models, 3 seeds, 2 widths, 2 learning rates, 2 weight decays, 3 runs, and 500 epochs would require about 14,400 model fits and should be treated as a full production experiment.

### Example Paper Run

```bash
python exps/symbolic_sequence_benchmark.py \
  --models LSTM GRU minGRU minLSTM Transformer LinearAttention Performer RWKV \
  --symbols 2 4 6 8 \
  --complexities 10 30 50 70 90 \
  --sequence-lengths 3500 \
  --window-size 100 \
  --forecast-horizon 100 \
  --layers 1 2 3 \
  --units 64 128 256 \
  --d-models 256 \
  --optimizers AdamW \
  --learning-rates 0.0001 0.0003 \
  --weight-decays 0.0 0.01 \
  --runs 3
```

For scaling-law-style sweeps, add token ratios. The script converts each ratio into a model-specific sequence length using `sequence_length = min(max_sequence_length, ceil(ratio * number_of_parameters))` and also keeps any fixed values passed through `--sequence-lengths`:

```bash
python exps/symbolic_sequence_benchmark.py \
  --models LSTM GRU minGRU minLSTM LinearAttention Performer RWKV \
  --token-ratios 1 5 20 \
  --max-sequence-length 20000
```

Results are written incrementally to `exps/results_symbolic/results.csv`, with the exact run configuration saved as `exps/results_symbolic/config.json`.

### Slurm on Convergence

The repository includes a Convergence-ready Slurm script:

```bash
sbatch scripts/run_symbolic_benchmark_slurm.sh
```

The script requests one `a100_3g.40gb` GPU, 12 CPU threads, 64 GB RAM, and runs an eight-way array with at most five tasks active at once. It writes one CSV shard per array task. The array job uses a filesystem lock so that only one task creates or repairs `.venv`; the other tasks wait until `.venv/.slearn_experiment_deps_ready` exists.

If an older job failed while creating `.venv`, resubmitting with the updated script is enough. To clean manually before resubmission, remove the incomplete environment and stale lock once from the login node:

```bash
rm -rf .venv .venv.lock
```

Merge the shards after completion with:

```bash
bash scripts/merge_symbolic_results.sh exps/results_symbolic/slurm_<array_job_id>
```

Generate publication-style figures locally after the CSV is available:

```bash
bash scripts/run_symbolic_visualizations.sh exps/results_symbolic/slurm_<array_job_id>/results_merged.csv
```

If `exps/results_symbolic/results_merged.csv` or `exps/results_symbolic/results.csv` exists, the script can also infer the input:

```bash
bash scripts/run_symbolic_visualizations.sh
```

The visualization script uses a fixed publication-style encoding for each model: color, marker shape, hollow/filled marker state, and line style are consistent across all generated figures. Legends are placed outside the axes at the bottom of each figure.

For figure-specific layout tuning, edit `FIGURE_LAYOUTS` in `exps/visualize_symbolic_results.py`; entries such as `rollout_error_vs_horizon`, `compute_performance_pareto`, `test_loss_vs_model_params`, and `dl_vs_model_params` can each set independent `figsize`, `legend_y`, `bottom`, and `legend_ncol` values.

The Slurm script can be configured through environment variables:

```bash
MODELS="LSTM GRU minGRU minLSTM Transformer LinearAttention Performer RWKV BERT GPT" \
MAX_EPOCHS=300 \
RUNS=3 \
SEED_COUNT=3 \
sbatch scripts/run_symbolic_benchmark_slurm.sh
```

### Local Visualization

Visualization is intentionally kept outside the Slurm job so that figures can be regenerated locally after inspecting or filtering the CSV. After a single-machine run, use:

```bash
python exps/visualize_symbolic_results.py \
  --results exps/results_symbolic/results.csv \
  --output-dir exps/figures_symbolic
```

After a Slurm array run, merge first and then plot:

```bash
bash scripts/merge_symbolic_results.sh exps/results_symbolic/slurm_<array_job_id>
python exps/visualize_symbolic_results.py \
  --results exps/results_symbolic/slurm_<array_job_id>/results_merged.csv \
  --output-dir exps/figures_symbolic/slurm_<array_job_id>
```

The script writes each analysis as a separate figure in PNG and PDF by default:

- next-token accuracy, cross-entropy, DL, and JW versus LZW complexity;
- cumulative rollout error versus forecast horizon;
- compute-performance Pareto plot;
- model-size and sequence-length scaling plots when the corresponding sweep exists;
- context-window sensitivity when multiple window sizes exist;
- model-by-complexity heatmaps.

All plots use shared font-size constants for axis labels, ticks, titles, annotations, and legends. Multi-model legends are placed outside the axes at the bottom center.

## Legacy Experiments

The older `exps/it_*_scale.py` and `exps/test_low_*.py` scripts are retained for reproducibility and now recognize the added model names. For new manuscript runs, prefer `exps/symbolic_sequence_benchmark.py` because it avoids target leakage in the Transformer decoder path and initializes rollouts from the observed prefix instead of the withheld target.

## Symbolic Time-Series Representation

`slearn` also contains SAX-style transforms and ABBA-related utilities for converting real-valued time series into symbolic sequences and reconstructing approximate signals.

```python
import numpy as np
from slearn.symbols import SAX

t = np.linspace(0, 10, 100)
ts = np.sin(t) + np.random.normal(0, 0.1, 100)

sax = SAX(window_size=10, alphabet_size=8)
symbols = sax.fit_transform(ts)
reconstruction = sax.inverse_transform()
```

## Distances

```python
from slearn.dmetric import (
    normalized_damerau_levenshtein_distance,
    normalized_jaro_winkler_distance,
)

dl = normalized_damerau_levenshtein_distance("ABBA", "ABAB")
jw = normalized_jaro_winkler_distance("ABBA", "ABAB")
print(dl, jw)
```

Both normalized distances are in `[0, 1]`; lower values indicate closer strings.

## Citation

If you use `slearn` or the LZW symbolic string library, please cite:

```bibtex
R. Cahuantzi, X. Chen, and S. Guettel, "A Comparison of LSTM and GRU Networks for Learning Symbolic Sequences," in Intelligent Computing, Springer Nature Switzerland, 2023, pp. 771-785.
```

If you use the ABBA-based symbolic prediction tools, please cite:

```bibtex
X. Chen, Fast Aggregation-Based Algorithms for Knowledge Discovery, Ph.D. dissertation, The University of Manchester, 2024.
```

## License

This project is licensed under the [MIT License](https://github.com/nla-group/slearn/blob/master/LICENSE).
