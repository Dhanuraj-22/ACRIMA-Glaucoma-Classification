# ACRIMA Glaucoma Classification Using Multi-Level Feature Fusion

## Overview

This project focuses on automated glaucoma classification from fundus images using deep learning.

The ACRIMA dataset is used for model development and evaluation. Four different deep learning architectures are investigated:

- ResNet50
- ConvNeXt-Tiny
- MaxViT-Tiny
- Swin-Tiny

The project first establishes baseline performance using the four backbone architectures without Multi-Level Feature Fusion (MLFF). MLFF is then introduced as the proposed feature-fusion approach to investigate whether combining features from different network levels improves glaucoma classification.

## Methodology

The project follows these main stages:

1. Train and evaluate four backbone architectures without MLFF.
2. Introduce Multi-Level Feature Fusion (MLFF) into each architecture.
3. Compare models with and without MLFF through an ablation study.
4. Perform external validation of the MLFF models using the RIM-ONE dataset.
5. Select the best-performing model based on internal and external evaluation.
6. Perform feature analysis and Explainable AI using Grad-CAM.
7. Investigate VLM-based analysis for additional interpretation.

## Dataset

### ACRIMA

The ACRIMA dataset contains 705 optic-disc fundus images:

- 396 glaucoma images
- 309 normal images

The images are divided using a stratified 80/20 split for the ablation experiments.

### RIM-ONE

RIM-ONE is used as an independent external validation dataset to evaluate the generalization capability of the proposed models.

## Models

### Baseline models

The following architectures are evaluated without MLFF:

- ResNet50
- ConvNeXt-Tiny
- MaxViT-Tiny
- Swin-Tiny

### Proposed models

MLFF is incorporated into each of the four architectures:

- ResNet50 + MLFF
- ConvNeXt-Tiny + MLFF
- MaxViT-Tiny + MLFF
- Swin-Tiny + MLFF

## Multi-Level Feature Fusion

MLFF combines feature representations from multiple levels of a deep network.

Lower-level features capture information such as edges and textures, while deeper levels capture higher-level structural and semantic information. The proposed fusion mechanism combines these representations before classification.

## Ablation Study

An ablation study is performed by comparing each backbone with and without MLFF.

The purpose is to determine whether MLFF contributes to the classification performance rather than assuming that the proposed module improves every architecture.

The detailed results are available in:

`results/ablation_without_mlff.csv`

and

`results/mlff_results.csv`

## External Validation

The MLFF models are evaluated on the RIM-ONE test set to assess their performance on an independent dataset.

The external validation results are available in:

`results/rimone_external_validation.csv`

## Explainable AI

Grad-CAM is used to visualize the regions of fundus images that contribute to the model's predictions.

Feature-level analysis is also performed to investigate the learned representations at different network levels and after feature fusion.

## Project Structure

```text
ACRIMA-Glaucoma-Classification/
│
├── notebooks/
│   ├── 01_ACRIMA_Without_MLFF_Baseline.ipynb
│   ├── 02_ACRIMA_With_MLFF_Proposed.ipynb
│   └── README.md
│
├── results/
│   ├── README.md
│   ├── ablation_without_mlff.csv
│   ├── mlff_results.csv
│   └── rimone_external_validation.csv
│
└── README.md
