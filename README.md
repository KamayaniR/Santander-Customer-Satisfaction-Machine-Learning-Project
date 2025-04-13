# Santander-Customer-Satisfaction-Machine-Learning-Project

## Overview
The objective of this project is to predict customer satisfaction levels using a highly anonymized tabular dataset provided by Santander Bank. The challenge involves:
 - Thousands of anonymized features
 - Strong class imbalance (only a small % of unsatisfied customers)
 - No business feature names or domain metadata
This project focuses on using a Decision Tree classifier for its transparency, simplicity, and ease of interpretation while exploring hyperparameter tuning and feature selection to improve performance and generalization.

## Key Learning Goals
 - Clean high-dimensional, noisy datasets
 - Perform EDA and variance-based feature filtering
 - Implement and tune Decision Tree classifiers
 - Evaluate models using AUC, F1-Score, and Kaggle leaderboard
 - Understand trade-offs in model complexity and interpretability
   
## Dataset Summary
 - train.csv: 76020 rows, 370 anonymized features + TARGET column
 - TARGET: Binary label (1 = unsatisfied customer)
 - Class distribution: ~96% satisfied, ~4% unsatisfied
 - Many columns with:
    - Zero variance
    - Sparse, non-informative distributions
  
 ## Workflow Summary
 ## 1. Exploratory Data Analysis (EDA)
  - Detected zero-variance features using VarianceThreshold
  - Identified 3 key relevant features:
    - var15 – Age-like variable
    - num_var45_ult1 – Activity/engagement
    - saldo_var30 – Possibly balance
  - Visualized feature correlation and class imbalance
  - Dropped features with no variation and no importance
    
 ## 2. Preprocessing
  - Split dataset into X (features) and y (TARGET)
  - No categorical encoding needed (data was numeric only)
  - Applied train_test_split (80/20) to create validation set
  - Standardized features for SVM/MLP (optional)
    
 ## 3. Model Development – Decision Trees
   ## Created multiple Decision Tree models with different hyperparameters:
    - Model               Parameters                             Validation AUC
    - Default Tree        No tuning                                 0.5600
    - Deep Tree           max_depth=10, min_samples_split=5         0.7900
    - Gini Tree           criterion='gini', max_depth=8             0.7970
    - Entropy Tree        criterion='entropy', max_depth=8          0.7900
  - Gini Tree found to generalize best, striking a balance between accuracy and overfitting.
    
## 4. Hyperparameter Tuning
   Used GridSearchCV to fine-tune:
   - criterion: 'gini' vs 'entropy'
   - max_depth: 4 to 12
   - min_samples_split: 2 to 10
     
## 5. Evaluation
   Metrics Used:
  - ROC-AUC: Primary metric for evaluation due to class imbalance
  - F1 Score: Assessed class 1 (unsatisfied) performance
  - Confusion Matrix: Checked true positive/false negative rates
    
## Kaggle Results & Model Comparison
 -  Model             Public Score       Private Score       Notes
 -  Default Tree      0.68872            0.67190             Overfitted/underfit
 -  Deep Tree         0.80227            0.79045             Slight overfitting
 -  Entropy Tree      0.80455            0.79048             Complex but stable
 -  Gini Tree         0.81054            0.79782             Best balance of generalization and simplicity


  








   




