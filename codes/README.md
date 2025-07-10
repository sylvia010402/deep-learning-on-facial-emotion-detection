# Code

This folder contains the complete Python pipeline for the Student Emotional Wellness Detection System, organized into modular scripts for reproducibility and educational technology deployment.

## Script Organization

- `1_data_preparation.py`  
  Handles all imports, file extraction, image loading, preprocessing, and data generator setup using TensorFlow and Keras.

- `2_model_building.py`  
  Contains the code for building and training multiple models including custom CNN and transfer learning architectures.

- `3_model_evaluation.py`  
  Includes test set evaluation, performance metrics (accuracy, F1-score), and visualizations like confusion matrices.



### Technologies Used

Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn



### Key Outcomes

- Custom CNN achieved 69% accuracy, outperforming larger pre-trained models.
- Grayscale inputs were sufficient for reliable emotional classification.
- The model showed clear strengths in identifying happy and surprise expressions, but struggled to separate sad and neutral.



This project is part of my data science portfolio, with a focus on applying machine learning to education and mental health. For more projects and insights, visit [my GitHub profile](https://github.com).
