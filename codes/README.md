# Code

This folder contains the complete Python pipeline for the Student Emotional Wellness Detection System, organized into modular scripts for reproducibility and educational technology deployment.

## Script Organization

**Core Pipeline**
- `data_preprocessing.py` - Image processing, augmentation, and data loader creation
- `exploratory_analysis.py` - Dataset visualization and class distribution analysis
- `model_training.py` - Complete training pipeline for all model architectures
- `model_evaluation.py` - Performance assessment and confusion matrix generation
- `utils.py` - Helper functions and configuration settings

**Model Architectures**
- `baseline_cnn.py` - Simple CNN baseline model
- `enhanced_cnn.py` - Complex custom CNN with 5 convolutional blocks  
- `transfer_learning.py` - VGG16, ResNet101, and EfficientNetB0 implementations

**Analysis and Visualization**
- `results_analysis.py` - Comparative performance analysis across models
- `visualization.py` - Training curves, confusion matrices, and error analysis plots

## Methodology Overview

The project follows a systematic approach to emotion recognition:

1. **Data Preparation**: Images preprocessed to 48x48 pixels with normalization and augmentation
2. **Architecture Comparison**: Testing both transfer learning (VGG16, ResNet101, EfficientNetB0) and custom CNN approaches
3. **Training Strategy**: Using callbacks for early stopping, learning rate reduction, and model checkpointing
4. **Evaluation**: Comprehensive assessment using accuracy, precision, recall, F1-scores, and confusion matrices
5. **Educational Focus**: Model selection optimized for deployment in resource-constrained educational environments

## Key Technical Decisions

- **Grayscale Processing**: Demonstrated equivalence to color processing while reducing computational overhead
- **Custom Architecture**: 5-block CNN with batch normalization, LeakyReLU, and strategic dropout placement
- **Transfer Learning Comparison**: Systematic evaluation of frozen vs. fine-tuned approaches
- **Educational Optimization**: Lightweight models suitable for real-time classroom deployment

## Running the Complete Pipeline

```bash
# 1. Explore and prepare data
python exploratory_analysis.py
python data_preprocessing.py

# 2. Train all model variants
python model_training.py

# 3. Evaluate and compare results
python model_evaluation.py
python results_analysis.py

# 4. Generate visualizations
python visualization.py
```

## Dependencies

```
tensorflow>=2.8.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
```

The pipeline is designed for educational technology applications, emphasizing interpretability, efficiency, and practical deployment considerations for student mental health monitoring systems.
