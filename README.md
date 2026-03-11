# BioHCI: A Python Framework for Physiological Signal Processing and Modeling

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
```

---

## Core Components
`BioHCI/architectures`

Contains implemented learning architectures and model wrappers, including modules such as:

- CNN
- LSTM
- CNN-LSTM
- 2D CNN-LSTM
- MLP
- RNN
- SVC wrapper
- VAE-based models
- Attention encoder-decoder variants

This structure supports experimentation across a broad range of sequence-modeling and classification approaches.

`BioHCI/data`

Handles subject-oriented dataset construction. Data are loaded into subject-level representations, which are then used for training, validation, and testing workflows.

`BioHCI/data_processing`

Contains preprocessing and feature engineering utilities, including balancing, statistical feature construction, wavelet-based processing, and within-subject oversampling.

`BioHCI/learning`

Implements analyzers, trainers, and evaluators used to run experiments and compare performance across settings.

`BioHCI/visualizers`

Provides utilities for plotting raw data, training behavior, and evaluation outputs.

`config_files`

Contains study-specific TOML files that define dataset locations, sensor metadata, relevant columns, chunk sizes, folds, and modeling settings.

---

## How the Framework Works

At a high level, a BioHCI experiment follows this workflow:

**1. Load a study configuration**

  A TOML file defines dataset and experiment parameters.

**2. Construct the dataset**

  Data are loaded and organized into subject-based structures.
    
**3. Preprocess the data**

  Depending on the study and configuration, preprocessing may include:

   - selecting relevant columns
   - standardization
   - chunking/windowing
   - feature construction
   - transforms such as wavelets
   - class balancing
  
**4. Split the data**

  The framework supports cross-subject and within-subject evaluation workflows.

**5. Define the model and training setup**

  A neural or classical model is configured and paired with the appropriate experiment settings.

**6. Train and evaluate**

  The analyzer coordinates training, validation, and testing, and performs extensive statistical analysis on the results.

**7. Save artifacts and visualize results**

  Outputs may include trained models, log files, plots, and confusion matrices.

---

## Entry Points
`BioHCI/main.py`

This is the primary experiment script. It supports the overall experiment workflow, including:

- parsing command-line options
- loading a study configuration
- constructing train / validation / test datasets
- preprocessing and feature construction
- defining the learning setup
- running evaluation
- logging and visualization

`BioHCI/single_sample_main.py`

This script is intended for single-sample inference or lightweight evaluation. It loads a configured dataset and saved model, then performs prediction on individual samples.

---

## Configuration System

One of the core design features of BioHCI is its use of study-specific TOML configuration files.

Configurations are stored in `config_files/` and are used to populate a `StudyParameters` object that controls experiment behavior.

### Example configuration

```text

resource_path = "CTS_CHI2020/tmp/tmp_train"
study_name = "CTS_CHI2020"
sensor_name = "Capacitor"
file_format = ".csv"
relevant_columns = "[0:192]"
start_row = 0
cat_names = "dir"
num_subj = 24
plot_labels = ["Voltage Frequency Gain", "Time"]
standardize = false
compute_fft = false
chunk_instances = true
samples_per_chunk = 250
interval_overlap = false
construct_features = true
feature_window = 250
feature_overlap = false
num_folds = 3
num_threads = 12
neural_net = true
classification = true

```

### Adding a New Study

To define a new study:
1. create a new TOML file in config_files/
2. specify the dataset path and study metadata
3. define relevant columns, labels, and chunking parameters
4. configure feature construction and model settings
5. point the entry script to the new configuration file

---

## Data Organization

BioHCI is organized around subject-level datasets.

The data loader expects data to be arranged so that each subject has its own subdirectory within a configured dataset root. These are then loaded into `Subject` objects and stored in dictionaries keyed by subject name. This organization decision is based on the knowledge that there are likely going to be similarities in the data generated from the same subject, and typically we cannot treat those signal measurements as independent.

A typical layout may look like:

```text
Resources/
└── <study_name>/
    ├── train/
    │   ├── subj1/
    │   ├── subj2/
    │   └── ...
    └── test/
        ├── subj1/
        ├── subj2/
        └── ...
```

Depending on the study, categories may be derived from directory names or filenames.

---

## Installation

This repository does not currently include a packaged installation file such as requirements.txt, setup.py, or pyproject.toml, so setup is manual.

1. Clone the Repository
  ```
  git clone https://github.com/denisaqori/BioHCI-Project.git
  cd BioHCI-Project
  ```
2. Create a Python Environment

  Using `conda`:
  ```
  conda create -n biohci python=3.10
  conda activate biohci
  ```

  Or using `venv`:
  ```
  python -m venv .venv
  source .venv/bin/activate
  ```
  On Windows:
  ```
  .venv\Scripts\activate`
  ```

3. Install Dependencies

  Based on the current codebase, you will likely need:
  ```
  pip install numpy scipy matplotlib seaborn toml torch scikit-learn
  ```

  Additional packages may be required depending on the specific modules or experiments you use.

4. Ensure the Repository Root Is on `PYTHONPATH`

  From the repository root:
  ```
  export PYTHONPATH=$PYTHONPATH:.
  ```
  On Windows PowerShell:
  ```
  $env:PYTHONPATH = "$env:PYTHONPATH;."
  ```

---

## Running an Experiment

From the repository root:
```
python BioHCI/main.py
```
To disable CUDA:
```
python BioHCI/main.py --disable-cuda
```
To enable visualization and verbose output:
```
python BioHCI/main.py --visualization --verbose
```
### Important Note

The main script currently expects a specific configuration file to be selected in code, so you may need to edit `main.py` and update the configuration filename manually.

---

## Running Single-Sample Evaluation

To run single-sample inference or evaluation:
```
python BioHCI/single_sample_main.py
```
This script expects:

- a valid study configuration
- a compatible dataset layout
- an existing saved model in the appropriate saved_objects/... directory

You may need to update the hardcoded configuration name and model path before use.

---

## Outputs

The framework supports saving and visualizing experiment artifacts such as:

- trained model checkpoints
- confusion matrices
- plots and visualizations
- logged evaluation outputs

These are typically written under saved_objects/ and related study-specific directories.

---

### Included Study Configurations

The repository contains multiple study-specific configuration files, including examples such as:

- `CTS_4Electrodes.toml`
- `CTS_5taps_per_button.toml`
- `CTS_CHI2020_train.toml`
- `CTS_EICS2020.toml`
- `CTS_Keyboard.toml`
- `CTS_Keyboard_simple.toml`
- `CTS_UbiComp2020.toml`
- `CTS_UbiComp2020_1sample.toml`
- `EEG_Workload.toml`
- `Solid_Pad_Gestures.toml`

These provide examples of how the framework can be adapted to different datasets and tasks.

--- 


