# Machine Learning Mastery with Python

## Description

This repository focuses on mastering Machine Learning using **Python**, following a structured and hands-on approach. The content is derived from the course [Machine Learning A-Z](https://www.udemy.com/course/machinelearning/) designed by industry experts, offering a detailed dive into essential Machine Learning concepts, algorithms, and techniques.

The objective of this project is to learn through the complexities of Machine Learning, starting from data preprocessing to advanced topics such as deep learning and model selection. By working through these materials, I will build a strong foundation in Python-based Machine Learning.

All code execution will be carried out in **Google Colab**, utilizing its powerful resources to ensure smooth and efficient computation within notebooks.

## Project Structure

The content is organized into ten main parts:

- **Part 1**: Data Preprocessing
- **Part 2**: Regression
  - Simple Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
  - Support Vector Regression (SVR)
  - Decision Tree Regression
  - Random Forest Regression
- **Part 3**: Classification
  - Logistic Regression
  - K-Nearest Neighbors (K-NN)
  - Support Vector Machines (SVM)
  - Kernel SVM
  - Naive Bayes
  - Decision Tree Classification
  - Random Forest Classification
- **Part 4**: Clustering
  - K-Means Clustering
  - Hierarchical Clustering
- **Part 5**: Association Rule Learning
  - Apriori Algorithm
  - Eclat Algorithm
- **Part 6**: Reinforcement Learning
  - Upper Confidence Bound (UCB)
  - Thompson Sampling
- **Part 7**: Natural Language Processing (NLP)
  - Bag-of-Words Model
  - NLP Algorithms
- **Part 8**: Deep Learning
  - Artificial Neural Networks (ANNs)
  - Convolutional Neural Networks (CNNs)
- **Part 9**: Dimensionality Reduction
  - Principal Component Analysis (PCA)
  - Linear Discriminant Analysis (LDA)
  - Kernel PCA
- **Part 10**: Model Selection & Boosting
  - k-fold Cross Validation
  - Hyperparameter Tuning
  - Grid Search
  - XGBoost

Each section operates independently, allowing me to either follow the project from start to finish or focus on specific topics as needed. This flexibility is essential in adapting my learning path to my career goals.

## Tools and Technologies

- **Python**: Core programming language for all machine learning implementations.
- **Google Colab**: For executing notebooks, providing cloud resources to streamline model training and testing.


## Environment Setup

To run the notebooks in **Google Colab**, follow these steps to set up the environment and access the necessary datasets stored in your Google Drive:

### 1. Upload Notebooks and Datasets
Ensure that your notebooks and datasets are uploaded to a folder in your Google Drive. For example, the folder path could be:

```
/MyDrive/Colab Notebooks/Machine Learning A-Z/
```

### 2. Mount Google Drive in Colab
To access the datasets from Google Drive, you will need to mount the drive in your Colab notebook. This can be done using the following code at the beginning of each of the notebooks:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Once your drive is mounted, you can navigate to the appropriate folder where your dataset is stored.

### 3. Loading the Dataset
After mounting the drive, load the dataset by specifying the correct file path. For example, to load the **Position_Salaries.csv** dataset for Polynomial Regression, use:

```python
import pandas as pd
dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Machine Learning A-Z/Part 2 - Regression/3 Polynomial Regression/Position_Salaries.csv')
```

### 4. Running the Notebook
Once the environment is set up, you can proceed with running each cell of the notebook in sequence. Colab provides GPU and TPU resources if needed for more computationally intensive tasks, such as training deep learning models.
