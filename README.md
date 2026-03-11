# BioHCI: A Python Framework for BioHCI Signal Processing and Modeling

A Python framework for configuring, preprocessing, modeling, and evaluating human-generated sequential data across BioHCI studies.

BioHCI is a Python framework for loading, preprocessing, visualizing, and modeling physiological and human-generated sequential data across BioHCI studies. It was designed to support end-to-end experimentation, from study configuration and dataset construction to feature extraction, model training, evaluation, and visualization.

The framework is structured to make it easier to run and compare experiments across multiple studies involving time-series and sensor-based data, including capacitive touch and other physiological or behavioral signals.

---

## Overview

BioHCI provides reusable infrastructure for:

- study-driven experiment configuration via TOML files
- subject-based dataset construction
- signal chunking and feature construction
- class balancing and oversampling
- cross-subject and within-subject evaluation workflows
- neural and non-neural modeling pipelines
- visualization of raw signals, learning curves, and confusion matrices
- experimentation across multiple sensor studies and label schemes

The repository is organized as a general experimentation framework for structured, human-generated sequential data, with support for both classical machine learning and deep learning workflows.

---

## Key Features

- **Configuration-driven experiments**  
  Study-specific settings are defined in TOML files, making it easy to reuse the same codebase across multiple datasets and tasks.

- **Reusable data pipeline**  
  The framework provides a consistent subject-oriented pipeline for loading, organizing, preprocessing, and splitting data.

- **Flexible preprocessing**  
  Supports operations such as column selection, chunking/windowing, feature construction, balancing, and optional transforms.

- **Multiple modeling approaches**  
  Includes neural architectures such as CNNs, LSTMs, CNN-LSTM variants, VAEs, and attention-based models, alongside wrappers for classical methods such as SVC.

- **Built-in evaluation and visualization**  
  Includes utilities for training analysis, testing, plotting, and confusion matrix generation.

- **Multi-study support**  
  The repository contains several example configuration files corresponding to different BioHCI studies and tasks.

---

## Repository Structure

```text

BioHCI-Project/
├── BioHCI/
│   ├── architectures/          # Neural network architectures and model wrappers
│   ├── data/                   # Subject loading, dataset construction, dataset splitting
│   ├── data_augmentation/      # Data augmentation and generative utilities
│   ├── data_processing/        # Feature construction, balancing, transforms
│   ├── definitions/            # Experiment and model definitions
│   ├── frequency_matching/     # Frequency-based matching utilities
│   ├── helpers/                # Utility and configuration helper functions
│   ├── knitted_components/     # Knitted sensor / component-related code
│   ├── learning/               # Training, evaluation, analyzers
│   ├── saved_objects/          # Saved models, confusion matrices, and artifacts
│   ├── visualizers/            # Visualization tools for signals and results
│   ├── main.py                 # Main experiment entry point
│   ├── single_sample_main.py   # Single-sample inference / evaluation script
│   ├── bode_analysis_stats.py
│   └── shazam_style_matching.py
├── config_files/               # Study-specific TOML configuration files
├── TestBioHCI/                 # Test scaffolding and related files
└── README.md
