# Colorizing Sketches with GANs — TensorFlow Image-to-Image Translation (Pix2Pix)

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/danishjavedcodes/colorizing-sketches)
[![Medium Article](https://img.shields.io/badge/Medium-Read%20Article-black?style=flat&logo=medium)](https://medium.com/towards-artificial-intelligence/colorising-sketches-using-generative-adversarial-networks-gans-e604842ba421)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://www.python.org/)

Train a **Generative Adversarial Network (GAN)** to **colorize black-and-white sketches** into full-color images. This project implements a **Pix2Pix-style** pipeline in **TensorFlow/Keras**: a **U-Net generator** paired with a **PatchGAN discriminator**, trained on the **Sketch2Pokemon** dataset (830 sketch–color pairs).

> **Full walkthrough:** [Colorizing sketches using GANs (Medium / Towards AI)](https://medium.com/towards-artificial-intelligence/colorising-sketches-using-generative-adversarial-networks-gans-e604842ba421)  
> **Runnable notebook:** [Colorizing Sketches on Kaggle](https://www.kaggle.com/code/danishjavedcodes/colorizing-sketches)

---

## Table of Contents

- [Overview](#overview)
- [Why This Project](#why-this-project)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Repository Contents](#repository-contents)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Training Details](#training-details)
- [Results](#results)
- [Learn More](#learn-more)
- [Author](#author)
- [Keywords](#keywords)

---

## Overview

**Sketch colorization** turns line-art or grayscale drawings into realistic color images—a core task in **computer vision**, **fashion design**, **animation**, and **game art pipelines**. Manual coloring is slow, expensive, and hard to scale.

This repository provides:

- **`colorizing-sketches.ipynb`** — end-to-end **GAN training** notebook (data loading → generator/discriminator → training → inference)
- Links to a **step-by-step Medium tutorial** and a **GPU-ready Kaggle notebook**

The model learns **image-to-image translation**: given a sketch (input), the generator predicts a colored image (target). Training uses **adversarial loss** plus **L1 (MAE) loss** for stable, sharp outputs—standard practice in **conditional GANs** and **pix2pix** systems.

---

## Why This Project

| Challenge | How GANs help |
|-----------|----------------|
| Hand-coloring is time-consuming | Automate colorization from paired sketch–photo data |
| Large-scale design variants need many artists | One trained model generates multiple colorizations |
| High production cost | Reuse a single model after training |

**Use cases:** concept art, comic inking, product mockups, Pokémon-style fan art, and prototyping **deep learning** pipelines before moving to diffusion models or larger datasets.

---

## Architecture

### Generator — U-Net

- **Encoder (downsampling):** 8 `Conv2D` blocks (64 → 512 filters), LeakyReLU, batch norm, dropout
- **Decoder (upsampling):** 7 `Conv2DTranspose` blocks with **skip connections** (U-Net)
- **Output:** 256×256×3 RGB, `tanh` activation

### Discriminator — PatchGAN

- Concatenates **input sketch** + **target or generated** image
- Classifies **local patches** (real vs. fake) instead of a single global score
- Improves texture and edge realism in generated colors

### Loss & optimization

| Component | Setting |
|-----------|---------|
| Generator loss | GAN loss + **λ × L1 loss** (λ = 100) |
| Discriminator loss | Binary cross-entropy (real vs. generated) |
| Optimizer | Adam, lr = **2e-4**, β₁ = 0.5 |
| Image size | **256 × 256** |
| Batch size | 2 |
| Epochs | **85** (checkpoints every 20 epochs) |

```
Sketch (input) ──► [ U-Net Generator ] ──► Colored image
                         ▲
                         │ feedback
              [ PatchGAN Discriminator ]
                   (sketch + image pair)
```

---

## Dataset

**[Sketch2Pokemon](https://www.kaggle.com/datasets/reiinakano/anime-sketch-colorization-pair)** on Kaggle — ~**830** paired images:

- Left half of each file: **sketch** (input)
- Right half: **colored** Pokémon-style artwork (target)

The notebook expects the unpacked layout:

```text
pokemon_pix2pix_dataset/
├── train/
│   └── *.jpg
└── test/
    └── *.jpg
```

On Kaggle, add the dataset to your notebook; paths are configured for `/kaggle/input/...`.

---

## Repository Contents

| File / folder | Description |
|---------------|-------------|
| [`colorizing-sketches.ipynb`](colorizing-sketches.ipynb) | Full **TensorFlow 2.x** implementation: preprocessing, U-Net, PatchGAN, training loop, visualization |
| [`README.md`](README.md) | Project documentation (this file) |

---

## Requirements

- **Python** 3.10+
- **TensorFlow** 2.x
- **NumPy**, **Matplotlib**, **pandas**
- **GPU** recommended (notebook tested on Kaggle **NVIDIA Tesla T4**)

```bash
pip install tensorflow numpy pandas matplotlib
```

---

## Quick Start

### Option A — Kaggle (recommended)

1. Open **[Colorizing Sketches on Kaggle](https://www.kaggle.com/code/danishjavedcodes/colorizing-sketches)**.
2. Add the **Sketch2Pokemon / anime sketch colorization** dataset.
3. Enable **GPU** accelerator.
4. Run all cells — training, checkpoints, and sample outputs are included.

### Option B — Local or Colab

1. Clone this repository:

   ```bash
   git clone https://github.com/danishjavedcodes/ColorizingSketches.git
   cd ColorizingSketches
   ```

2. Download and extract the [Sketch2Pokemon dataset](https://www.kaggle.com/datasets/reiinakano/anime-sketch-colorization-pair).
3. Update `PATH` in the notebook to your dataset root.
4. Open `colorizing-sketches.ipynb` in Jupyter, VS Code, or Google Colab.
5. Run training (`fit(train_dataset, EPOCHS, test_dataset)`).

Saved artifacts (from the notebook):

- **Checkpoints** — `tf.train.CheckpointManager` (every 20 epochs)
- **Generator weights** — `model_weights.h5`

---

## Training Details

**Preprocessing pipeline:**

1. Split combined JPG into sketch (left) and color (right)
2. Resize to 256×256 (test set)
3. Random crop (training)
4. Normalize pixels to **[-1, 1]** via `(x / 127.5) - 1`

**Training loop:**

- Custom `@tf.function` `train_step` alternates generator and discriminator updates
- After each epoch, visualizes **input | ground truth | prediction** on a test batch
- Final evaluation runs on **10 test images**

For theory, diagrams, and code explanations, see the **[Medium article](https://medium.com/towards-artificial-intelligence/colorising-sketches-using-generative-adversarial-networks-gans-e604842ba421)**.

---

## Results

After **~85 epochs** on Sketch2Pokemon, the generator produces plausible colorizations: learned hues, shading, and character-consistent palettes from sketch inputs alone.

Example layout from the notebook (each row: **sketch → ground truth → model output**):

| Input sketch | Ground truth | GAN prediction |
|:------------:|:------------:|:--------------:|
| *(see Kaggle notebook outputs)* | | |

Open the **[Kaggle notebook](https://www.kaggle.com/code/danishjavedcodes/colorizing-sketches)** or the **[Medium post](https://medium.com/towards-artificial-intelligence/colorising-sketches-using-generative-adversarial-networks-gans-e604842ba421)** for full result figures.

---

## Learn More

- **Tutorial:** [Colorizing sketches using Generative Adversarial Networks (GANs)](https://medium.com/towards-artificial-intelligence/colorising-sketches-using-generative-adversarial-networks-gans-e604842ba421) — *Towards AI*, Danish Javed et al.
- **Code:** [Kaggle: colorizing-sketches](https://www.kaggle.com/code/danishjavedcodes/colorizing-sketches)
- **Related ideas:** [pix2pix paper](https://arxiv.org/abs/1611.07004) (image-to-image translation), conditional GANs, U-Net segmentation backbones

**Suggested GitHub topics:** `gan`, `generative-adversarial-network`, `sketch-colorization`, `image-to-image-translation`, `pix2pix`, `tensorflow`, `computer-vision`, `deep-learning`, `unet`, `patchgan`, `pokemon`, `kaggle`

---

## Author

**Danish Javed** — AI/ML Engineer  

- GitHub: [@danishjavedcodes](https://github.com/danishjavedcodes)
- Medium: [@danish_javed](https://medium.com/@danish_javed)
- Kaggle: [danishjavedcodes](https://www.kaggle.com/danishjavedcodes)

If this project helped you, consider **starring the repo** and sharing the [Medium article](https://medium.com/towards-artificial-intelligence/colorising-sketches-using-generative-adversarial-networks-gans-e604842ba421) or [Kaggle notebook](https://www.kaggle.com/code/danishjavedcodes/colorizing-sketches).

---

## Keywords

`sketch colorization` · `colorizing sketches` · `GAN` · `generative adversarial network` · `Pix2Pix` · `U-Net generator` · `PatchGAN discriminator` · `TensorFlow Keras` · `image-to-image translation` · `conditional GAN` · `Sketch2Pokemon` · `anime sketch colorization` · `deep learning computer vision` · `Kaggle notebook` · `Pokémon sketch dataset`
