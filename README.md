# FaciesGAN: Multi-Scale Generative Adversarial Network for Geological Facies Prediction

## 📌 Overview
FaciesGAN is a specialized generative adversarial network designed to synthesize high-resolution 2D and 3D geological facies realizations. It operates on a pure-PyTorch, numeric-only pipeline, learning complex geological patterns from training datasets to generate realistic facies images conditioned on well log data and rock physics attributes.

## ✨ Key Features
* **Numeric-Only Pipeline**: Optimized for high-fidelity `.npy` data handling, bypassing image artifacts and quantization issues.
* **Multi-scale Progressive Training**: Learns geological features across a pyramid of scales, from coarse structural trends to fine-scale heterogeneities.
* **Parallel Training Groups**: Accelerates training by processing multiple pyramid scales simultaneously in parallel groups.
* **Multi-Channel Conditioning**: Integrates well log data and continuous rock physics properties (Ip, Is, Vp/Vs) as conditioning constraints.
* **Spatially-Adaptive Denormalization (SPADE)**: Uses SPADE modules to inject noise and conditioning data effectively across spatial dimensions.
* **Categorical One-Hot Logic**: Native support for one-hot encoded facies classes, ensuring sharp boundaries and correct class distributions.
* **Type-Safe & Modern**: Built with Python 3.10+, full type hints, and optimized PyTorch 2.x features (including `torch.compile` support).

## 🏗️ Architecture Components

### Core Modules
* **`models/`** - Pure PyTorch implementation of the FaciesGAN architecture.
    - `models/facies_gan.py` - Core GAN orchestration and optimization logic.
    - `models/generator.py` - Multi-scale generator with SPADE normalization.
    - `models/discriminator.py` - Multi-scale PatchGAN-style discriminator.
    - `models/custom_layer.py` - Optimized building blocks (ConvBlocks, SPADE, Minibatch StdDev).
* **`datasets/`** - High-performance numeric data loading.
    - `datasets/dataset.py` - Multi-scale numeric pyramid dataset.
    - `datasets/data_prefetcher.py` - Non-blocking GPU data transfer.
* **`training/`** - Training orchestration.
    - `training/trainer.py` - Parallel multi-scale training runner.
* **`interpolators/`** - Advanced numeric interpolation strategies for pyramid generation:
  - `NearestInterpolator`: Standard nearest-neighbor resampling for categorical data.
  - `RockPhysicsInterpolator`: Physics-aware interpolation for continuous attributes using Lanczos and Backus averaging.
  - `NeuralSmoother`: Deep learning-based interpolation for high-fidelity transitions.

### Data Structure
```
FaciesGAN/
├── data/
│   ├── facies/              # Categorical facies data (.npy)
│   ├── wells/               # Well log binary masks and mapping (.npz)
│   └── rock_physics/        # Continuous seismic attributes (Ip, Is, Vp/Vs) (.npy)
├── models/                  # PyTorch model definitions
├── datasets/                # Numeric data loading and prefetching
├── training/               # Training runners and group orchestration
├── interpolators/           # Multi-scale resampling strategies
├── results/                 # Training outputs, checkpoints, and logs
├── main.py                  # CLI entry point for training
├── gen_facies.py            # Realization generation script
├── experiments.py           # Ablation studies and benchmarking
└── options.py               # Centralized hyperparameter configuration
```

## 🚀 Installation

### Prerequisites
* Python 3.10+
* PyTorch 2.x with CUDA support
* NVIDIA GPU (12GB+ VRAM recommended for parallel training)

### Setup
```sh
git clone https://github.com/mazzutti/FaciesGAN.git
cd FaciesGAN
pip install -r requirements.txt
```

## 🎯 Quick Start

### Training FaciesGAN
Train the model on numeric data with parallel scale processing:
```sh
python3 main.py --input-path data --num-iter 2000 --batch-size 40 \
    --num-train-pyramids 200 --num-facies-classes 4 \
    --use-wells --use-rock-physics --num-parallel-scales 7
```

### Key Training Parameters:
- `--num-facies-classes`: Number of one-hot encoded facies categories (default: 4).
- **Default Facies Classes**:
  1. **Floodplain** (0): Light Green
  2. **Point bar** (1): Yellow
  3. **Channel** (2): Blue
  4. **Boundary** (3): Dark Gray
- `--num-parallel-scales`: Number of scales to train concurrently (default: 7).
- `--use-wells`: Enable well-log conditioning.
- `--use-rock-physics`: Enable multi-channel rock physics attribute conditioning.
- `--num-iter`: Optimization steps per scale (default: 2000).
- `--compile-backend`: Enable `torch.compile` for faster iteration speeds.

### Generating Realizations
Generate new conditional realizations from a trained model:
```sh
python3 gen_facies.py --how-many 500 \
    --model-path results/py/2025_04_26_facies_gan \
    --out-path results/generated --plot-well-mask
```

## 🔬 Technical Details

### SPADE-Based Conditioning
FaciesGAN utilizes Spatially-Adaptive Denormalization (SPADE) to inject well and rock-physics constraints into the generator. Unlike simple concatenation, SPADE learns to modulate feature maps at every spatial location, preserving sharp geological boundaries and honoring local constraints more effectively.

### Multi-Scale Parallelism
The project implements a "Group Trainer" strategy. Instead of training one scale at a time (standard progressive growth), FaciesGAN can train a block of scales simultaneously. This significantly reduces total training time while maintaining stable convergence by allowing gradients to flow through multiple resolutions at once.

### High-Fidelity Interpolation
For the numeric pipeline, simple bilinear interpolation is insufficient. The `RockPhysicsInterpolator` uses Lanczos resampling combined with Backus-style averaging logic to ensure that upsampled continuous properties (like acoustic impedance) maintain physically consistent values across scales.

## 📈 Recent Improvements
* **Project Flattening**: Streamlined codebase by removing legacy abstractions and multi-framework support logic for a faster PyTorch core.
* **Numeric-Only Workflow**: Completely eliminated PNG-based intermediate processing in favor of high-precision `.npy` tensors.
* **Optimized Trainer**: Improved DDP (Distributed Data Parallel) stability and NVLink-aware batching.
* **Rock Physics Integration**: Native support for Ip, Is, and Vp/Vs multi-channel conditioning.

---
*Developed as part of Postdoctoral research on Geological Modeling and Generative AI.*
