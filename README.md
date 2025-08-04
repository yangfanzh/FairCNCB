# Fair Graph Neural Networks with Counterfactual Augmentation

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-orange)
![PyG](https://img.shields.io/badge/PyG-2.0%2B-red)

A framework for training fair graph neural networks using counterfactual data augmentation to mitigate bias in node classification tasks.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Overview

This repository implements FairCNCB (Fair Graph Neural Network with Counterfactual and Category Balance), a novel approach for training fair GNNs that:

1. Generates counterfactual nodes using GANs to balance sensitive attribute distributions
2. Preserves graph topology while augmenting the dataset
3. Maintains model utility while improving fairness metrics

The method is evaluated on credit risk assessment datasets with sensitive attributes like gender and marital status.

## Key Features

- **Counterfactual Generation**: GAN-based generation of nodes with flipped sensitive attributes
- **Topology Preservation**: k-NN graph construction maintains original graph structure
- **Fairness Metrics**: Comprehensive evaluation including:
  - Statistical Parity Difference (SPD)
  - Equal Opportunity Difference (EOD)
  - Group-specific accuracy/F1 scores
- **Modular Architecture**: Easily extensible to different GNN backbones and datasets

## Datasets

Preprocessed versions of the following datasets are included:

1. **German Credit** (`German.csv`)
   - Task: Credit risk assessment
   - Sensitive attribute: Gender (Male/Female)
   - Nodes: 1,000
   - Features: 20 (mixed numeric/categorical)

2. **Credit Default** (`credit.csv`)
   - Task: Default prediction
   - Sensitive attribute: Married (Yes/No)
   - Nodes: 30,000 
   - Features: 23

3. **Bail Decision** (`bail.csv`)
   - Task: Bail grant prediction
   - Sensitive attribute: Race (White/Non-white)
   - Nodes: 18,876
   - Features: 18

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fair-gnn.git
cd fair-gnn
