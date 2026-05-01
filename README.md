# FaciesGAN

> **Multi-Scale Generative Adversarial Network for Physics-Informed Geological Facies Prediction**

FaciesGAN is a research-grade deep learning framework that synthesizes high-resolution 2D geological facies realizations conditioned on well-log data and seismic rock-physics attributes. It is built on a pure-PyTorch, numeric-only pipeline, supports multi-GPU distributed training (DDP/NVLink), and ships with a complete ablation-study experiment runner.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Repository Layout](#repository-layout)
5. [Installation](#installation)
6. [Data Preparation](#data-preparation)
7. [Training](#training)
   - [Single-GPU](#single-gpu-training)
   - [Multi-GPU (DDP)](#multi-gpu-ddp-training)
   - [Resuming & Fine-Tuning](#resuming--fine-tuning)
   - [Full Training Options Reference](#full-training-options-reference)
8. [Monitoring with TensorBoard](#monitoring-with-tensorboard)
9. [Generating Realizations](#generating-realizations)
10. [Reproducing the Experiments](#reproducing-the-experiments)
    - [Conditioning Ablation Suite](#conditioning-ablation-suite)
    - [Experiment Options Reference](#experiment-options-reference)
11. [Output Structure](#output-structure)
12. [Performance Tips](#performance-tips)
13. [Citation](#citation)
14. [License](#license)

---

## Overview

FaciesGAN learns the spatial statistics of geological facies from training images and generates new, statistically consistent realizations at multiple resolutions. The model operates on a **multi-scale pyramid** of numeric (`.npz`) data instead of raw images, enabling:

* Lossless, quantization-free representation of discrete facies classes and continuous physical properties.
* Physics-informed training with seismic rock-physics constraints (Acoustic Impedance **Ip**, Shear Impedance **Is**, Vp/Vs ratio).
* Well-log conditioning that spatially constrains the generator to honor borehole observations.

---

## Key Features

| Feature | Description |
|---|---|
| **Numeric-Only Pipeline** | All I/O uses `.npz` tensors — no PNG round-trips or quantization artifacts |
| **Multi-Scale Progressive Growth** | Coarse-to-fine pyramid training from 12 px up to 1024 px |
| **Parallel Scale Groups** | Train multiple pyramid scales simultaneously for faster convergence |
| **SPADE Conditioning** | Spatially-Adaptive Denormalization injects well-log and rock-physics constraints |
| **One-Hot Facies Encoding** | Sharp class boundaries and correct categorical distributions |
| **Rock-Physics Branch** | Generator predicts Ip, Is, Vp/Vs channels alongside facies |
| **Physics Loss** | Synthetic seismic forward-modeling loss (Ricker wavelet convolution) |
| **DDP / NVLink Support** | Full `torchrun` multi-GPU training with NCCL |
| **`torch.compile` Support** | Inductor backend for Ampere+ GPU acceleration |
| **AMP Training** | Mixed-precision with `bf16`/`fp16` autocast |
| **Gradient Checkpointing** | Reduces peak VRAM at the cost of ~30 % extra compute |
| **TensorBoard Logging** | Per-scale loss curves, facies grids, and rock-physics visualizations |
| **Ablation Runner** | Automated 4-variant conditioning experiment with embedding analysis |

---

## Architecture

```
                        Noise (z)
                           │
                    ┌──────▼──────┐
  Wells Pyramid ───►│             │
                    │  Generator  │◄──── SPADE Normalization
Seismic Pyramid ───►│  (Multi-    │      (wells + rock-physics)
                    │   Scale)    │
                    └──────┬──────┘
                           │
              ┌────────────┴───────────────┐
              │                            │
       Facies Output               Rock-Physics Output
     (one-hot, C classes)         (Ip  │  Is  │  Vp/Vs)
              │                            │
       ┌──────▼──────┐             ┌──────▼──────┐
       │ PatchGAN    │             │  Physics    │
       │ Discrimin.  │             │  Loss (SEG) │
       └─────────────┘             └─────────────┘
```

### Core Modules

| Module | Description |
|---|---|
| `models/facies_gan.py` | GAN orchestration, noise amplitude scheduling, multi-scale forward pass |
| `models/generator.py` | Multi-scale SPADE generator |
| `models/discriminator.py` | Multi-scale PatchGAN discriminator |
| `models/custom_layer.py` | ConvBlocks, SPADE, Minibatch StdDev |
| `models/base.py` | Loss functions: WGAN-GP, Dice, Well, TV, Elastic, Physics |
| `training/trainer.py` | Parallel group trainer — coordinates scale groups and DDP ranks |
| `datasets/dataset.py` | `TorchPyramidsDataset` — multi-scale numeric pyramid loader |
| `datasets/data_prefetcher.py` | Non-blocking GPU data prefetcher |
| `interpolators/` | Nearest (categorical), rock-physics (Lanczos + Backus), neural smoother |
| `physics/` | Ricker wavelet synthesis, rock-physics forward model |
| `tensorboard_visualizer.py` | Async background plotter for TensorBoard |
| `experiments/` | 4-variant ablation runner with manifold embedding analysis |

---

## Repository Layout

```
FaciesGAN/
├── data/                        # Training data (npz)
│   ├── facies.npz               # Categorical facies (N, H, W)
│   ├── wells.npz                # Well-log conditioning masks
│   ├── vp.npz                   # P-wave velocity volume
│   ├── vs.npz                   # S-wave velocity volume
│   ├── rho.npz                  # Density volume
│   ├── seismic.npz              # Observed seismic volume
│   └── stats.json               # Per-channel normalization statistics
│
├── models/                      # PyTorch model definitions
├── datasets/                    # Data loading & prefetching
├── training/                    # Trainer & DDP orchestration
├── interpolators/               # Multi-scale resampling strategies
├── physics/                     # Seismic forward-modeling
├── experiments/                 # Ablation runner & plotting
│
├── main.py                      # ★ Training entry point
├── resume.py                    # Resume / fine-tune from checkpoint
├── gen_facies.py                # ★ Post-training generation script
├── generate_report.py           # Report generation helper
├── plot_pyramids.py             # Pyramid visualization utility
├── options.py                   # TrainingOptions & ResumeOptions dataclasses
├── config.py                    # Global constants (paths, filenames)
├── tensorboard_visualizer.py    # Async TensorBoard helper
├── background_workers.py        # Thread-pool plot submission
├── launch_tensorboard.sh        # TensorBoard launcher script
├── requirements.txt             # Runtime dependencies
└── requirements-dev.txt         # Development / lint dependencies
```

---

## Installation

### Prerequisites

* **Python** ≥ 3.10
* **PyTorch** ≥ 2.0 with CUDA support
* **NVIDIA GPU** — 12 GB VRAM minimum; 24 GB+ recommended for parallel-scale training

### Setup

```bash
git clone https://github.com/mazzutti/FaciesGAN.git
cd FaciesGAN

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install runtime dependencies
pip install -r requirements.txt

# (Optional) Install development / type-checking tools
pip install -r requirements-dev.txt
```

> **Note on NVIDIA Apex**: `requirements.txt` lists `nvidia-apex`. If your environment does not support it, remove the line and training will fall back to native PyTorch AMP. Apex is used only for additional optimizer fused kernels.

---

## Data Preparation

FaciesGAN expects data in the `data/` directory as compressed NumPy archives. The expected files and shapes are:

| File | Shape | Description |
|---|---|---|
| `facies.npz` | `(N, H, W)` uint8 | Integer class labels (0 … C-1) |
| `wells.npz` | `(N, C, H, W)` float32 | One-hot well-log conditioning masks |
| `vp.npz` | `(N, H, W)` float32 | P-wave velocity (m/s) |
| `vs.npz` | `(N, H, W)` float32 | S-wave velocity (m/s) |
| `rho.npz` | `(N, H, W)` float32 | Density (kg/m³) |
| `seismic.npz` | `(N, H, W)` float32 | Observed seismic amplitude |
| `stats.json` | — | Per-channel mean/std used for normalization |

The default facies class mapping (editable in `models/palette.py`) is:

| Class ID | Lithofacies | Colour |
|---|---|---|
| 0 | Floodplain | Light green |
| 1 | Point bar | Yellow |
| 2 | Channel | Blue |
| 3 | Boundary | Dark gray |

Set `--num-facies-classes` to match the number of classes in your dataset.

---

## Training

### Single-GPU Training

```bash
python3 main.py \
    --input-path data \
    --output-path outputs/py \
    --num-train-pyramids 200 \
    --batch-size 40 \
    --num-iter 2000 \
    --num-parallel-scales 7 \
    --num-facies-classes 4 \
    --use-wells \
    --use-seismic \
    --use-rock-physics
```

The script automatically detects the GPU via `--gpu-device` (default: 0) and creates a timestamped output directory under `--output-path`.

### Multi-GPU (DDP) Training

Use `torchrun` for distributed training across multiple GPUs on the same node:

```bash
# 2-GPU training with NVLink
NCCL_P2P_LEVEL=NVL \
NCCL_ALGO=Ring \
NCCL_PROTO=Simple \
OMP_NUM_THREADS=4 \
torchrun --nproc_per_node=2 main.py \
    --input-path data \
    --output-path outputs/py \
    --num-train-pyramids 200 \
    --batch-size 40 \
    --num-workers 4 \
    --num-iter 2000 \
    --num-parallel-scales 7 \
    --use-wells \
    --use-seismic \
    --use-rock-physics \
    2>&1 | tee outputs/train.log
```

> **Tip**: Visualization and checkpoint writes happen only on **rank 0**. All other ranks participate in collective operations but produce no I/O.

### Resuming & Fine-Tuning

To resume an interrupted training run from the last saved checkpoint:

```bash
python3 resume.py \
    --checkpoint-path outputs/py/2025_04_26_10_30_00 \
    --num-parallel-scales 7
```

To fine-tune a fully trained model with additional iterations:

```bash
python3 resume.py \
    --fine-tuning \
    --checkpoint-path outputs/2025_04_26_10_30_00 \
    --num-iter 500 \
    --start-scale 4
```

The resume script reads the original `options.json` saved in the checkpoint directory, so all original hyperparameters are preserved automatically.

### Full Training Options Reference

#### Data & I/O

| Flag | Default | Description |
|---|---|---|
| `--input-path` | `data` | **Required.** Path to dataset root directory |
| `--output-path` | `outputs/py` | Base output directory (a timestamp subfolder is created automatically) |
| `--output-fullpath` | `None` | Override the full output path (no timestamp added) |
| `--num-facies-classes` | `4` | Number of discrete facies classes (one-hot channels) |
| `--noise-channels` | `3` | Number of noise channels injected per scale |
| `--crop-size` | `256` | Spatial crop size used during training |
| `--num-train-pyramids` | `200` | Number of training pyramid samples |
| `--num-workers` | `auto` | DataLoader workers (default: `min(4, cpu_count//2)`) |
| `--regen-npy-gz` | `False` | Force regeneration of cached pyramid files |
| `--manual-seed` | `None` | Fixed random seed for reproducibility |
| `--no-shuffle` | — | Disable dataset shuffling |

#### Model Architecture

| Flag | Default | Description |
|---|---|---|
| `--num-features` | `32` | Base feature count in the first network layer |
| `--min-num-features` | `32` | Minimum feature count across all layers |
| `--kernel-size` | `3` | Convolution kernel size |
| `--num-layers` | `5` | Number of layers per scale block |
| `--stride` | `1` | Convolution stride |
| `--padding-size` | `0` | Convolution padding size |

#### Pyramid & Scale

| Flag | Default | Description |
|---|---|---|
| `--stop-scale` | `6` | Final scale index (number of pyramid levels - 1) |
| `--start-scale` | `0` | Scale to start/resume training from |
| `--start-epoch` | `0` | Epoch to resume from within the current scale group |
| `--min-size` | `12` | Minimum spatial size at the coarsest pyramid scale |
| `--max-size` | `1024` | Maximum spatial size |
| `--num-parallel-scales` | `2` | Number of scales trained simultaneously |

#### Optimization

| Flag | Default | Description |
|---|---|---|
| `--num-iter` | `2000` | Optimization steps (full dataset passes) per scale |
| `--batch-size` | `1` | Per-GPU batch size |
| `--lr-g` | `5e-4` | Generator learning rate |
| `--lr-d` | `5e-4` | Discriminator learning rate |
| `--beta1` | `0.5` | Adam β₁ |
| `--gamma` | `0.9` | Discriminator StepLR decay factor |
| `--lr-decay` | `1000` | Epochs between discriminator LR decay steps |
| `--lr-decay-unit` | `epoch` | LR decay unit: `epoch`, `step`, or `batch` |
| `--lr-patience` | `400` | Generator ReduceLROnPlateau patience |
| `--lr-min` | `1e-4` | Minimum generator LR |
| `--lr-smoothing-alpha` | `0.95` | EMA factor for generator loss fed to scheduler |
| `--lr-g-factor` | `0.8` | Generator LR reduction factor on plateau |
| `--generator-steps` | `3` | Generator inner steps per iteration |
| `--discriminator-steps` | `3` | Discriminator inner steps per iteration |
| `--scale0-disc-steps-multiplier` | `1` | Extra D-steps multiplier at scale 0 |
| `--grad-clip-norm` | `1.0` | Max gradient norm for generator clipping (0 = disabled) |
| `--gp-interval` | `16` | Lazy gradient-penalty interval (StyleGAN2 style) |

#### Loss Weights

| Flag | Default | Description |
|---|---|---|
| `--facies-rec-loss-penalty` | `10.0` | Facies reconstruction (Dice) loss weight |
| `--well-loss-penalty` | `10.0` | Well-log conditioning loss weight |
| `--gradient-loss-penalty` | `0.1` | Discriminator gradient-penalty weight |
| `--adversarial-loss-penalty` | `1.0` | Generator adversarial loss weight |
| `--diversity-loss-penalty` | `1.0` | Generator diversity loss weight |
| `--num-diversity-samples` | `3` | Noise samples per G-step for diversity loss |
| `--rec-rock-physics-loss-penalty` | `1.0` | Rock-physics reconstruction loss weight |
| `--tv-loss-penalty` | `1.0` | Total-variation smoothness loss weight (rock physics) |
| `--elastic-loss-penalty` | `1.0` | Elastic consistency loss weight (Ip/Is vs Vp/Vs) |
| `--physics-loss-penalty` | `1.0` | Seismic physics loss weight (forward model MSE) |
| `--scale0-loss-multiplier` | `1.0` | Extra loss multiplier for scale 0 |

#### Conditioning

| Flag | Description |
|---|---|
| `--use-wells` | Enable well-log conditioning |
| `--use-seismic` | Enable seismic data loading |
| `--use-rock-physics` | Enable rock-physics branch (Ip, Is, Vp/Vs outputs + losses) |
| `--wells-mask-columns` | Explicit well column indices to use (space-separated integers) |

#### Seismic / Rock-Physics Physics

| Flag | Default | Description |
|---|---|---|
| `--dz-pixel` | `5.0` | Vertical resolution in metres per pixel |
| `--wavelet-f-peak` | `8.0` | Ricker wavelet peak frequency (Hz) |
| `--wavelet-dt` | `0.001` | Wavelet sampling interval (s) |

#### Hardware & Performance

| Flag | Default | Description |
|---|---|---|
| `--gpu-device` | `0` | GPU device ID (single-GPU mode) |
| `--use-cpu` | — | Force CPU training |
| `--amp-dtype` | `bf16` | AMP compute dtype: `bf16` (Ampere+) or `fp16` |
| `--compile-backend` | — | Enable `torch.compile` (Inductor) |
| `--no-compile` | — | Disable `torch.compile` even on Ampere+ |
| `--gradient-checkpoint` | — | Activation checkpointing (~30 % slower, lower peak VRAM) |
| `--use-profiler` | — | Export a Chrome trace via PyTorch Profiler |

#### Logging & Output

| Flag | Default | Description |
|---|---|---|
| `--save-interval` | `100` | Epochs between saving generated output grids |
| `--checkpoint-interval` | `1` | Epochs between saving training-state checkpoints |
| `--num-real-facies` | `5` | Real facies rows in the output grid |
| `--num-generated-per-real` | `5` | Generated columns per real facies in the grid |
| `--no-tensorboard` | — | Disable TensorBoard logging |
| `--no-plot-outputs` | — | Disable PNG sample plots during training |

---

## Monitoring with TensorBoard

A convenience launcher script is provided that kills any previous instance and starts TensorBoard pointing at `outputs/`:

```bash
# Default: logdir=outputs/, port=6006
./launch_tensorboard.sh

# Custom logdir and port
./launch_tensorboard.sh outputs/py/my_run 6007
```

Then open **http://localhost:6006** in your browser.

TensorBoard tracks:

* **Losses** — Discriminator, Generator, Reconstruction, Well, Rock-Physics, Physics (per scale)
* **Facies grids** — Real vs. generated facies at each scale
* **Rock-physics grids** — Ip, Is, Vp/Vs per scale
* **Seismic diagnostic** — Synthetic vs. real seismic at each scale
* **Learning rates** — Generator and discriminator LR per scale

---

## Generating Realizations

After training, generate new conditional realizations with `gen_facies.py`:

```bash
# Basic: generate 500 realizations
python3 gen_facies.py \
    --how_many 500 \
    --model_path outputs/py/2025_04_26_10_30_00 \
    --out_path outputs/generated

# With well-mask overlay
python3 gen_facies.py \
    --how_many 200 \
    --model_path outputs/py/2025_04_26_10_30_00 \
    --plot_well_mask

# Compare real vs. generated (3 variants per real, 5 real images)
python3 gen_facies.py \
    --how_many 1 \
    --model_path outputs/py/2025_04_26_10_30_00 \
    --comparison_plots \
    --num_generated 3 \
    --num_real 5
```

### Manifold / Latent Space Analysis

Visualize the distributional match between real and generated facies in latent space:

```bash
python3 gen_facies.py \
    --how_many 500 \
    --model_path outputs/py/2025_04_26_10_30_00 \
    --plot_mds \
    --plot_umap \
    --plot_isomap \
    --plot_tsne
```

### `gen_facies.py` Options Reference

| Flag | Description |
|---|---|
| `--how_many` | **Required.** Number of realizations to generate |
| `--model_path` | **Required.** Path to trained model checkpoint directory |
| `--out_path` | Output directory (default: same as `--model_path`) |
| `--rec` | Generate reconstruction sample (same size as training image) |
| `--gpu_device` | GPU device ID (default: 0) |
| `--use_gpu` | Explicitly enable GPU |
| `--wells` | Well indices for conditioning (default: 0–199) |
| `--plot_well_mask` | Overlay well mask on generated facies plots |
| `--comparison_plots` | Generate real-vs-generated comparison grids |
| `--num_generated` | Variants per real facies in comparison grids (default: 3) |
| `--num_real` | Real facies rows in comparison grids (default: 5) |
| `--plot_scale` | Pyramid scale for comparison plots (default: finest) |
| `--plot_mds` | Save MDS embedding plot |
| `--plot_umap` | Save UMAP embedding plot |
| `--plot_isomap` | Save Isomap embedding plot |
| `--plot_tsne` | Save t-SNE embedding plot |

Generated realizations are saved as `.tif` files in `<out_path>/generated/`.

---

## Reproducing the Experiments

The `experiments/` package provides a fully automated conditioning-ablation suite that trains **four model variants** and produces comparison grids plus manifold embedding plots for all of them.

### Conditioning Ablation Suite

The ablation varies the **conditioning inputs** (Wells and Seismic) while keeping the **Rock-Physics output branch** (Ip, Is, Vp/Vs) always enabled — it is a network output, not a conditioning signal.

| Variant | Wells *(input)* | Seismic *(input)* | Rock-Physics *(output)* |
|---|:---:|:---:|:---:|
| `wells_seismic` | ✅ | ✅ | ✅ |
| `wells_only` | ✅ | ❌ | ✅ |
| `seismic_only` | ❌ | ✅ | ✅ |
| `unconditional` | ❌ | ❌ | ✅ |

> **Note**: Rock-Physics (Ip, Is, Vp/Vs) is always predicted as an **output** of the generator alongside the facies classes. Enabling `--use-rock-physics` activates the physics-informed loss terms that supervise these output channels during training.

#### Full Experiment (Train + Generate + Embed)

```bash
python3 -m experiments \
    --input-path data \
    --output-path outputs/experiments \
    --num-iter 2000 \
    --num-train-pyramids 200 \
    --batch-size 50 \
    --num-parallel-scales 7 \
    --stop-scale 6 \
    --nproc-per-node 2 \
    --how-many 2000 \
    --use-rock-physics \
    --embedding-methods isomap mds tsne umap \
    --embedding-data facies rock_physics
```

Each variant is trained in sequence using `torchrun` internally. The runner **automatically resumes** if a variant was partially trained — no manual intervention needed.

#### Generation Only (Models Already Trained)

```bash
python3 -m experiments \
    --input-path data \
    --output-path outputs/experiments \
    --skip-training \
    --model-paths \
        outputs/experiments/wells_seismic \
        outputs/experiments/wells_only \
        outputs/experiments/seismic_only \
        outputs/experiments/unconditional \
    --how-many 2000 \
    --embedding-methods isomap mds tsne umap
```

#### Disable Embedding Computation

```bash
python3 -m experiments \
    --input-path data \
    --output-path outputs/experiments \
    --no-embeddings \
    --skip-training \
    --model-paths ...
```

### Experiment Options Reference

| Flag | Default | Description |
|---|---|---|
| `--input-path` | — | **Required.** Dataset root directory |
| `--output-path` | `outputs/experiments` | Base output directory for all variants |
| `--how-many` | `2000` | Realizations to generate per variant |
| `--skip-training` | — | Skip training; use existing model paths |
| `--model-paths` | — | 4 model paths (requires `--skip-training`) |
| `--nproc-per-node` | `2` | GPUs for DDP per variant training |
| `--num-iter` | `2000` | Training iterations per scale |
| `--num-train-pyramids` | `10` | Training pyramids per variant |
| `--batch-size` | `50` | Per-GPU batch size |
| `--num-parallel-scales` | `7` | Parallel scales per variant |
| `--stop-scale` | `6` | Final scale index |
| `--use-rock-physics` | — | Enable rock-physics branch for all variants |
| `--embedding-methods` | all 4 | Manifold methods: `isomap mds tsne umap` |
| `--embedding-data` | `facies rock_physics` | Data types for embedding plots |
| `--embedding-per-facies` | — | Plot separate embeddings per conditioning crossline |
| `--no-embeddings` | — | Skip all latent-space visualization |
| `--manual-seed` | `None` | Fixed seed for reproducibility |
| `--gpu-device` | `0` | Primary GPU |
| `--compile-backend` | — | Enable `torch.compile` for variant training |
| `--no-compile` | — | Disable `torch.compile` for variant training |
| `--checkpoint-interval` | `1` | Checkpoint save interval (epochs) |

All other training hyper-parameters (`--lr-g`, `--lr-d`, `--gamma`, `--discriminator-steps`, etc.) are forwarded unchanged to each `main.py` invocation.

---

## Output Structure

A training run produces the following layout:

```
outputs/py/<timestamp>/
├── options.json                  # All training options (used by resume / gen_facies)
├── log.txt                       # Full training log
│
├── 0/                            # Scale 0 (coarsest)
│   ├── generator.pth             # Trained generator weights
│   ├── discriminator.pth         # Trained discriminator weights
│   ├── noise_amp.txt             # Noise amplitude for this scale
│   ├── shape.pth                 # Spatial shape descriptor
│   ├── rec_noise.pth             # Reconstruction noise
│   ├── masks.pth                 # Well-log masks
│   ├── epoch_checkpoint.pth      # Resumable optimizer/scheduler state
│   ├── completed_epoch.txt       # Last completed epoch marker
│   └── real_x_generated_facies/  # PNG grids (real vs. generated)
│       └── gen_0_<epoch>.png
│
├── 1/ … 6/                       # Scales 1–6 (same structure)
│
└── runs/                         # TensorBoard event files
    └── faciesgan/
        └── events.out.tfevents.*
```

---

## Performance Tips

* **Ampere+ GPUs (RTX 3090, A100, H100)**: Enable `bf16` (default) and `--compile-backend` for the best throughput.
* **Memory-constrained GPUs (≤ 12 GB)**: Use `--gradient-checkpoint` to reduce peak VRAM. This disables `torch.compile` automatically.
* **Large datasets**: Increase `--num-workers` (up to `cpu_count // 2`) and use `--batch-size` ≥ 8 for better GPU utilization.
* **DDP NVLink**: Set `NCCL_P2P_LEVEL=NVL NCCL_ALGO=Ring NCCL_PROTO=Simple` for optimal inter-GPU bandwidth.
* **Reproducibility**: Pass `--manual-seed <int>` to fix all random seeds.

---

## Citation

If you use FaciesGAN in your research, please cite:

```bibtex
@misc{mazzutti2025faciesgan,
  title   = {FaciesGAN: A Physics-Informed Multi-Scale GAN for Geological Facies Modeling},
  author  = {Mazzutti, Alessandro},
  year    = {2025},
  note    = {Postdoctoral research, Geological Modeling and Generative AI}
}
```

---

## License

This project is licensed under the terms of the [MIT License](LICENSE.md).

---

*Developed as part of Postdoctoral research on Geological Modeling and Generative AI.*
