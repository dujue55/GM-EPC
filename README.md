# GM-EPC: Gated Multimodal Emotion Prediction in Conversation

This repository contains the implementation and experimental code for the paper  
**“A Gated Multimodal Framework for Emotion Prediction in Conversation (GM-EPC)”**.

## Project Structure

The codebase is organized into two main parts:

- src/: Core implementation, including dataset handling, feature extraction, model definition, and training logic.  
  These modules were developed and tested locally.

- notebooks/: Experiment scripts intended to run in a GPU-enabled environment (e.g., Kaggle).  
  The notebook calls the functions implemented in src/ to reproduce the experiments.

## Directory Overview

notebooks/  
  main-experiment.ipynb   # Main experiment notebook (GPU / Kaggle)

src/  
  dataset.py              # Dataset loading and preprocessing  
  features.py             # Speech and text feature extraction  
  model.py                # GM-EPC model definition  
  trainer.py              # Training and evaluation logic  
  utils/                  # Utility functions   

requirements.txt          # Dependencies for Kaggle / GPU environment  
requirements-local.txt    # Dependencies for local development  

## Dataset

This work uses the IEMOCAP dataset, which is publicly available on Kaggle: https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease

Due to licensing restrictions, the dataset itself is not included in this repository.

## Feature Cache

The file features_cache_backup_new.zip contains cached feature representations extracted from the IEMOCAP dataset.  
This cache is provided for reproducibility and efficiency purposes.
## Environment Setup

For local development and debugging:
pip install -r requirements-local.txt

For GPU-based experiments (e.g., Kaggle):
pip install -r requirements.txt

