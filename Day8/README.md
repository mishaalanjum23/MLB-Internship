# Breast Cancer Prediction System

## What I Learned

In this project, I learned how to evaluate a classification model using Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix. These metrics help understand how well the model performs instead of relying only on accuracy.

## Hyperparameter Tuning

I learned that hyperparameter tuning is used to find the best settings for a model before training it. I used GridSearchCV, which tested different combinations of parameters using cross-validation and selected the best one.

## Best Parameters

The best parameters found by GridSearchCV were:

* **C:** 0.1
* **Solver:** liblinear

## Baseline vs Tuned Model

I first trained a baseline Logistic Regression model and then trained a tuned model using GridSearchCV. The tuned model performed better than the baseline and achieved higher Accuracy, Precision, Recall, and F1-Score.

## Key Observations

The tuned model made fewer incorrect predictions than the baseline model. This showed that choosing better hyperparameters can improve the model's overall performance.
