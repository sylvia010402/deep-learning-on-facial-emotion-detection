# Documentation

This folder contains all project documentation, technical reports, and data resources for the Student Emotional Wellness Detection System.

## Contents

**Technical Documentation**
- `technical_report.pdf` - Complete analysis methodology, model architecture details, and performance evaluation
- `data_codebook.md` - Dataset structure, variable definitions, and preprocessing steps
- `model_documentation.md` - CNN architecture specifications and training parameters

**Data Files**
- `emotion_dataset/` - Processed facial expression images organized by emotion category
- `training_validation_splits.csv` - Data partition information for reproducibility
- `preprocessing_parameters.json` - Image normalization and augmentation settings


## Data Overview

The emotion dataset consists of 15,109 grayscale facial images (48x48 pixels) categorized into four emotion classes relevant for educational monitoring:
- Happy: 3,976 images
- Sad: 3,982 images  
- Neutral: 3,978 images
- Surprise: 3,173 images

Images are preprocessed and normalized for consistent model input, with separate train/validation/test splits to ensure robust evaluation.
