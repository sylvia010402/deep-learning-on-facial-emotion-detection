# Results

This folder contains all outputs generated from the emotion recognition analysis, including model performance metrics, visualizations, and trained model files.

## Key Findings Summary

**Best Performing Model**: Custom CNN achieved 68.75% test accuracy with 0.68 loss
**Transfer Learning Results**: VGG16 (53%), ResNet101 & EfficientNetB0 (25%)
**Strongest Predictions**: Surprise (F1: 0.87) and Happy (F1: 0.78)
**Challenge Area**: Distinguishing between Sad and Neutral emotions

## Output Files

**Model Performance**
- `model_comparison_results.csv` - Accuracy and loss metrics for all tested architectures
- `confusion_matrices/` - Visual confusion matrices for each model
- `classification_reports/` - Detailed precision, recall, and F1-scores by emotion class
- `training_history/` - Loss and accuracy curves during model training

**Visualizations**
- `training_curves/` - Training and validation performance over epochs
- `error_analysis_plots/` - Misclassification patterns and model interpretability
- `data_distribution_charts/` - Class balance and sample image visualizations

**Model Assets**
- `best_model.h5` - Trained custom CNN model file
- `model_weights/` - Saved weights for reproducibility
- `preprocessing_pipeline.pkl` - Image preprocessing parameters

**Educational Applications**
- `demo_predictions/` - Sample emotion predictions on test images
- `real_time_demo_output/` - Screenshots and results from live emotion detection
- `performance_benchmarks.md` - Speed and accuracy metrics for deployment planning

## Using the Results

The trained model (`best_model.h5`) can be loaded directly for making predictions on new facial images. See the inference demo in the code folder for implementation examples. Performance visualizations provide insights into model behavior and can inform deployment decisions for educational technology applications.
