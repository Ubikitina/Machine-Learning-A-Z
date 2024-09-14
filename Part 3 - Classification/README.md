# Classification Models

In this section, we will focus on **Classification Models**, which are used to predict categorical outcomes. Unlike regression, which predicts continuous values, classification assigns data points to one of several predefined categories or classes. 

Classification is a supervised learning algorithm with numerous applications, including churn modeling in business (predicting customer retention), email classification (sorting emails as important, promotional, or spam), and image recognition (e.g., distinguishing between dogs and cats). 

![](img/Classification%20Model%20Applications.png)

By the end of this section, you will be familiar with a variety of classification techniques, and you'll be able to apply them to real-world problems requiring categorical predictions.

### Overview of Classification Models:

We will cover the implementation and interpretation of the following classification models:

- **Logistic Regression**: A linear model used for binary classification tasks. Despite its name, it is a classification algorithm that estimates the probability of a binary outcome.
- **K-Nearest Neighbors (K-NN)**: A non-parametric, instance-based learning algorithm that classifies a data point based on the majority class of its nearest neighbors.
- **Support Vector Machine (SVM)**: A powerful linear classifier that aims to find the hyperplane that best separates the classes.
- **Kernel SVM**: Extends SVM by using kernel tricks to handle non-linear classification problems.
- **Naive Bayes**: A probabilistic classifier based on Bayes' Theorem, assuming independence between predictors.
- **Decision Tree Classification**: A tree-like model used to make decisions based on a series of questions about the features.
- **Random Forest Classification**: An ensemble learning method that builds multiple decision trees and combines them to improve accuracy and reduce overfitting.

### Use Cases:
- **Binary vs. Multi-class Classification**: These models can be applied to both binary (two categories) and multi-class (more than two categories) classification problems. Examples include spam detection (binary) and image recognition (multi-class).
