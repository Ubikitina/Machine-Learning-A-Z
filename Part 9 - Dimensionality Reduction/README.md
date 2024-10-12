# Dimensionality Reduction

## Introduction

In Part 3 - Classification, we worked with datasets that contained only two independent variables. We did this for two main reasons:

1. **Visualization**: Two dimensions allow us to visualize the workings of machine learning models effectively. We can plot prediction regions and decision boundaries for each model, enhancing our understanding.

2. **Dimensionality Reduction**: Regardless of the original number of independent variables, we can often reduce them to two through appropriate dimensionality reduction techniques.
de

In this section we will look at different techniques for dimensionality reduction.

## Dimensionality Reduction Techniques
There are two types of Dimensionality Reduction techniques:

- Feature Selection

- Feature Extraction

### Feature Selection

Feature Selection techniques aim to identify and retain the most relevant features while eliminating those that are redundant or irrelevant. The following methods are commonly used:

- **All-in approach** (no dimensionality reduction at all): This method involves including all variables in the model from the start. While it’s quick, it can lead to overfitting and may incorporate variables that do not contribute meaningfully to predictions.

- **Backward Elimination**: This systematic approach removes unimportant variables iteratively, ensuring that only the most relevant variables are retained. This helps reduce the risk of overfitting.

- **Forward Selection**: This method starts with no variables in the model and adds them one by one based on their statistical significance. Although it’s effective for building models with many potential predictors, it can be more time-consuming than backward elimination.

- **Bidirectional Elimination (Stepwise Regression)**: This technique combines both backward elimination and forward selection. At each step, it evaluates the possibility of adding or removing variables, striving to balance between overfitting and underfitting. While thorough, it can be tedious (It is one of the most tedious).

- **All Possible Models - Score Comparison**: This approach evaluates various combinations of variables based on performance metrics (e.g., adjusted R², AIC, or BIC) to identify the most optimal subset of variables by comparing multiple models.

These techniques area already described in more detail in [Part 2 - Regression/2 Multiple Linear Regression/multiple_linear_regression.ipynb](../Part%202%20-%20Regression/2%20Multiple%20Linear%20Regression/multiple_linear_regression.ipynb).

### Feature Extraction

Feature Extraction is a set of techniques used to transform raw data into a reduced set of features that retain the essential information needed for analysis and modeling. Unlike Feature Selection, which focuses on identifying and selecting a subset of the existing features, Feature Extraction creates new features from the original data.

Feature Extraction transforms the original features into new ones (e.g., principal components in PCA). These new features are often combinations of the original features, which can make it challenging to interpret their significance or relate them back to the original data.

In this section (Part 9), we will cover the following Feature Extraction techniques:

- **Principal Component Analysis (PCA)**: A technique that transforms the data into a set of orthogonal (uncorrelated) variables called principal components, which capture the maximum variance.

- **Linear Discriminant Analysis (LDA)**: A technique used to find a linear combination of features that best separate two or more classes.

- **Kernel PCA**: An extension of PCA that uses kernel methods to allow for non-linear dimensionality reduction.
