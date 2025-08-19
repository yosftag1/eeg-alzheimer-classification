# EEG Graph Classification for Alzheimer's Detection

> Thesis note: This work was completed for my graduation thesis. Most experiments were run on Kaggle and organized as Jupyter notebooks, so the bulk of the code lives in notebooks throughout this repo.

## Data

- Kaggle dataset (prepared/uploaded version): https://www.kaggle.com/datasets/yosftag/open-nuro-dataset
- Original OpenNeuro dataset: https://openneuro.org/datasets/ds004504/versions/1.0.6

This repository contains the implementation of a graph neural network-based approach for EEG signal classification, specifically designed for Alzheimer's disease detection using the OpenNeuro dataset.

## Project Overview

This project explores neural‑network–based EEG classification for early detection of Alzheimer's disease, leveraging graph neural networks (GNNs), transformer models, and convolutional neural networks (CNNs). The pipeline combines time–frequency analysis, connectivity measures, and graph-based representations of brain activity.

## Repository Structure

```
├── ARCHIVE_SUMMARY.md
├── EXECUTION_GUIDE.md
├── README.md
├── requirements.txt
│
├── 01_data_preprocessing/             # Data cleaning and preprocessing
│   └── data_cleaning_mne.ipynb        # MNE-based EEG data cleaning
├── 02_feature_extraction/             # Feature extraction from EEG signals
│   └── feature_extraction_main.ipynb  # Main feature extraction pipeline
├── 03_time_frequency_analysis/        # Time–frequency domain analysis
│   ├── continuous_wavelet_transform.ipynb  # CWT generation and analysis
│   ├── spectrograms_analysis.ipynb         # Spectrogram analysis
│   └── time_frequency_transforms.ipynb     # Various time–frequency transforms
├── 04_graph_construction/             # Graph construction and GNN inputs
│   └── eeg_graph_construction.ipynb   # EEG graph construction (e.g., EEGRAPH)
├── 05_model_training/                 # Model training implementations
│   ├── cwt_training.ipynb                  # CWT-based model training
│   ├── main_training_loop.ipynb            # Main training loop pipeline
│   ├── transformer_training.ipynb          # Transformer-based model training
│   └── transformer_seb_training.ipynb      # Transformer with SE blocks
├── 06_hyperparameter_optimization/    # Hyperparameter tuning
│   └── hyperparameter_optimization.ipynb   # Optimization for graph models
├── 07_evaluation/                     # Model evaluation and testing
│   ├── model_testing.ipynb                 # Model evaluation and testing
│   └── pretrained_evaluation.ipynb         # Pre-trained model evaluation
└── notebooks_archive/                 # Archive of experimental/duplicate notebooks
```

## Key Features

### 1. Data Preprocessing
- EEG data cleaning using MNE-Python
- Artifact removal and signal filtering (missing here as i didnt creat that part)
- Data epoching and preprocessing for analysis

### 2. Feature Extraction
- Spectral coherence calculation
- Relative Band Power (RBP) computation
- Connectivity measures across EEG channels
- Graph-based feature extraction

### 3. Time-Frequency Analysis
- Continuous Wavelet Transform (CWT) generation
- Spectrogram analysis
- Multi-scale time-frequency representations
- Band-specific frequency analysis (delta, theta, alpha, beta, gamma)

### 4. Graph Neural Networks
- EEG signal to graph conversion using EEGRAPH library
- Multiple connectivity measures:
  - Pearson correlation
  - Squared coherence
  - Cross-correlation
  - Weighted Phase Lag Index (wPLI)
  - Shannon entropy
- Graph-based classification models

### 5. Deep Learning Models
- Transformer-based architectures for EEG classification
- Convolutional neural networks for time-frequency data
- Graph neural networks for connectivity-based features

## Setup

Install dependencies with the pinned requirements file:

```
pip install -r requirements.txt
```

### Key Dependencies
- **MNE-Python**: EEG data processing and analysis
- **PyTorch**: Deep learning framework
- **EEGRAPH**: EEG to graph conversion
- **SciPy**: Signal processing utilities
- **Scikit-learn**: Machine learning utilities

## Usage

### Quickstart

- Option A — Kaggle (recommended if you already use Kaggle):
  - Create a Kaggle Notebook and attach the dataset: https://www.kaggle.com/datasets/yosftag/open-nuro-dataset
  - Upload or import the notebook you want to run (see Workflow below) and execute cells top-to-bottom.

- Option B — Local:
  - Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
  - Launch Jupyter and open the target notebook:
   ```bash
   jupyter lab
   ```

### Workflow (recommended order)
1) Preprocess raw EEG with MNE:
  - `01_data_preprocessing/data_cleaning_mne.ipynb`
2) Extract features from cleaned data:
  - `02_feature_extraction/feature_extraction_main.ipynb`
3) Generate time–frequency representations (optional but useful):
  - `03_time_frequency_analysis/continuous_wavelet_transform.ipynb`
  - `03_time_frequency_analysis/spectrograms_analysis.ipynb`
4) Build EEG graphs for GNNs:
  - `04_graph_construction/eeg_graph_construction.ipynb`
5) Train models:
  - `05_model_training/main_training_loop.ipynb`
  - `05_model_training/transformer_training.ipynb`
  - `05_model_training/cwt_training.ipynb`
6) Evaluate:
  - `07_evaluation/model_testing.ipynb`

## Methodology

### Signal Processing Pipeline
1. **Preprocessing**: Raw EEG signals are cleaned and filtered
2. **Feature Extraction**: Multiple features extracted including:
   - Spectral power in different frequency bands
   - Connectivity measures between channels
   - Time-frequency representations
3. **Graph Construction**: EEG channels represented as nodes, connectivity as edges
4. **Classification**: Graph neural networks trained for binary classification

### Model Architectures
- **Graph Neural Networks**: For connectivity-based features
- **Transformers**: For sequential EEG data
- **CNNs**: For time-frequency representations (spectrograms, CWT)

## Results and Performance

The models are evaluated using:
- Classification accuracy
- Precision, recall, and F1-score
- ROC-AUC analysis
- Cross-validation performance


## Contributing

This repository represents my thesis research work. For questions inquiries, please open an issue or email me at yosftag2000@gmail.com .

## License

This project is part of academic research. Please cite appropriately if using any part of this code for research purposes.

