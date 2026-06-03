# SnnForMI

English | [Russian](README_RU.md)

SnnForMI is a research codebase for EEG-based motor imagery classification. It focuses on lightweight spiking neural networks for compact and energy-aware inference, while keeping conventional EEG baselines in the same experimental environment for comparison.

The repository was reorganized around a clear experiment pipeline: raw EEG data is downloaded and preprocessed in `src/data`, model architectures live in `src/models`, and training or tuning scripts are grouped under `src/training`. Root-level scripts remain as compatibility launchers, so commands such as `python main.py ...` still work.

## Reference Work

The main reference for the lightweight spiking model is:

> H. Zhang, H. Wang, J. An, S. Zheng, and D. Wu, "A lightweight spiking neural network for EEG-based motor imagery classification," Key Laboratory of the Ministry of Education for Image Processing and Intelligent Control, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan 430074, China.


Implementation notes:

- Reference implementation used for orientation: [`Zhr1110/SnnForMI`](https://github.com/Zhr1110/SnnForMI). That repository is the public implementation accompanying the lightweight SNN paper and contains the original `preprocess.py`, `preprocess_for_filterbank.py`, `main.py`, and `CUPY_SNN_PLIF`-based experiment structure.
- The reference `CUPY_SNN_PLIF` implementation follows the PLIF-style neuron design used in the lightweight SNN paper.
- CuPy-backed PLIF/LIF neuron modules are implemented in `tools/neurons.py`.
- Project SNN architectures are implemented in `src/models/SNNs.py`.
- SpikingJelly is used as an implementation reference for neuron dynamics, surrogate-gradient training, step-mode utilities, and CUDA-oriented spiking modules.
- Additional baselines include EEGNet, ShallowConvNet, DeepConvNet, FBCNet, and a Norse latency-coded SNN.

Compared with the reference repository, this version reorganizes the code into `src/data`, `src/models`, and `src/training`, keeps root-level compatibility wrappers, adds additional evaluation utilities, and records local benchmark artifacts for thesis experiments.

## Benchmark Snapshot

The following values are taken from stored benchmark artifacts in `benchmarks/`. They should be read as project experiment records rather than universal leaderboard scores.

| Dataset / setting | Model | Accuracy | Macro-F1 | Source |
|---|---:|---:|---:|---|
| BNCI2014-001 | LIF readout, no subject tuning | 69.10% | 69.03% | `full_eval_lif_readout_best_seed26.json` |
| BNCI2014-001 | LIF readout, subject tuning | 69.98% | 69.91% | `full_eval_subject_tuning_lif_seed26_selected.json` |
| BNCI2014-001 | LIF readout, best merged selection | 71.06% | 71.01% | `full_eval_subject_tuning_lif_seed26_final_merged.json` |
| BNCI2014-001 | Lightweight SNN reference row | 68.92% | n/a | `bnci2014_001_accuracy_f1_plot_data.csv` |
| Weibo2014 | LIF readout, no tuning | 54.30% | 54.08% | `full_eval_weibo2014_lif_readout_best_seed28.json` |
| Weibo2014 | LIF readout, subject tuning | 54.56% | 54.40% | `full_eval_weibo2014_lif_seed28_combined_tune.json` |
| Weibo2014 | LIF readout, target merged selection | 55.57% | 55.46% | `full_eval_weibo2014_lif_seed28_target_merged.json` |
| Weibo2014 | Lightweight SNN reference row | 56.64% | n/a | `weibo2014_accuracy_f1_plot_data.csv` |

Efficiency snapshot for BNCI2014-001:

| Model | Trainable parameters | Energy / sample | Avg. power | Inference time |
|---|---:|---:|---:|---:|
| `CUPY_SNN_PLIF` | 2,601 | 0.825 mJ | 82.299 W | 0.02599 s |
| `CUPY_SNN_LIF_READOUT` | 2,609 | 1.038 mJ | 82.299 W | 0.03268 s |
| `EEGNet` | 3,476 | 2.748 mJ | 161.946 W | 0.04398 s |
| `ShallowConvNet` | 46,084 | 47.466 mJ | 113.738 W | 1.08172 s |
| `DeepConvNet` | 320,479 | 45.571 mJ | 167.663 W | 0.70451 s |

## Project Layout

- `src/data/`: dataset downloaders, preprocessing pipelines, Euclidean alignment, filterbank preparation, and the PyTorch dataset wrapper.
- `src/models/`: EEGNet, ConvNet, FBCNet, Norse, and CuPy-based SNN model architectures.
- `src/training/`: main training entry points, grid runners, subject tuning scripts, and dataset-specific training scripts.
- `tools/`: low-level spiking backend, CUDA helpers, surrogate gradients, augmentations, and compatibility wrappers.
- `benchmarks/results/`: curated JSON/CSV benchmark summaries used in the README tables and plotting scripts.
- `benchmarks/figures/`: generated figures grouped by benchmark family.
- `benchmarks/runs/`: local raw run logs and scratch experiment traces; these are kept out of normal Git history.
- `data/`: raw and preprocessed dataset files.

## Data Preparation

Prepare a single-band EEG representation:

```bash
python preprocess.py --dataset 0 --resample_fs 250
```

Prepare filterbank data for FBCNet:

```bash
python preprocess_for_filterbank.py --dataset 2 --resample_fs 250
```

Dataset ids:

- `0`: BNCI2014001
- `1`: BNCI2014002
- `2`: Weibo2014

## Training

Run the main within-subject training pipeline:

```bash
python main.py --dataset 0 --model CUPY_SNN_LIF_READOUT --subject_id 1 --trial_num 1 --epoch 1500 --epoch2 600 --batch_size 128 --device 0
```

Run a full grid over selected variants:

```bash
python run_full_grid.py --dataset 0 --model CUPY_SNN_LIF_READOUT --epoch 1500 --epoch2 600 --batch_size 128 --device 0
```

Run a smoke grid:

```bash
python run_full_grid.py --smoke --subjects 1 --variants aug_base
```

## Notes

The repository is organized for thesis experiments rather than as a packaged library. Some CuPy-based SNN models require a CUDA-capable environment and the corresponding CuPy build. CPU-only imports of baseline models are supported through lazy model loading in `src/models`.
