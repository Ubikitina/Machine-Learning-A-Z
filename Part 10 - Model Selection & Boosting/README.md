# Model Selection & Boosting

## Introduction

After building our Machine Learning models, several critical questions remain:

1. **Bias-Variance Tradeoff**: The bias-variance tradeoff is the balance between a model’s simplicity (high bias leads to underfitting) and its flexibility (high variance leads to overfitting). How do we balance the bias-variance tradeoff when constructing and evaluating our models?
2. **Hyperparameter Optimization**: How can we determine the optimal values for hyperparameters (those parameters that are not learned directly by the model, e.g. max depth of trees, number of neurons, learning rates, etc.)?
3. **Model Selection**: How can we identify the most appropriate Machine Learning model for a specific business problem?


## Model Selection & Boosting Techniques

In this part, we will address these questions using Model Selection techniques, including:

- **k-Fold Cross Validation**: A technique that allows us to assess the performance of our model by splitting the dataset into k folds and training the model k times, each time using a different fold as the validation set.
- **Grid Search**: A method for hyperparameter tuning that tests multiple combinations of hyperparameters to find the optimal configuration for our model.

## XGBoost (Bonus)

Finally, we conclude this part with a bonus section dedicated to the powerful and increasingly popular **XGBoost** Machine Learning model. We will explore its implementation and understand why it is so effective for a variety of tasks.
