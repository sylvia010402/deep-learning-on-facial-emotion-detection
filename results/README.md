# Results

This folder contains all outputs generated from the emotion recognition analysis, including model performance metrics, visualizations, and trained model files.

## Key Findings Summary

**Best Performing Model**: Custom CNN achieved 68.75% test accuracy with 0.68 loss
**Transfer Learning Results**: VGG16 (53%), ResNet101 & EfficientNetB0 (25%)
**Strongest Predictions**: Surprise (F1: 0.87) and Happy (F1: 0.78)
**Challenge Area**: Distinguishing between Sad and Neutral emotions

## Output Files

- `distribution_of_classes.png`
  Dataset numnber of distributions of each emotion.
  
- `confusion_matrix_model3.png`  
  A heatmap showing the final model’s prediction accuracy across all emotion classes.

- `classification_report_model3.txt`  
  A summary of precision, recall, and F1-scores for happy, sad, neutral, and surprise.

- `accuracy_loss_curves_model3a.png`  
  Training and validation accuracy/loss across 35 epochs for the custom CNN.

- `accuracy_loss_curves_model3b.png`  
  Training and validation accuracy/loss across 35 epochs for the custom CNN.

- `model_comparison_table.png`  
  A table summarizing test accuracy and performance insights for VGG16, ResNet101, EfficientNetB0, and the custom model.


## Key Takeaways

- The custom CNN consistently outperformed all transfer learning models with a test accuracy of approximately 69%.
- Happy and surprise expressions were most accurately predicted, while sad and neutral were often confused.
- The validation loss remained stable, indicating minimal overfitting and good generalization.
