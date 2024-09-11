# Naive Bayes Classifier Implementation

## Overview

This repository contains implementations of a Naive Bayes Classifier for binary classification tasks using both a custom implementation and the `scikit-learn` library. The focus is on the classification of breast cancer data from the Breast Cancer Wisconsin dataset.

## Contents

- `naive_bayes_classifier_from_scratch.py`: Custom implementation of Naive Bayes Classifier from scratch without using libraries.
- `naive_bayes_classifier_sklearn.py`: Implementation of Naive Bayes Classifier using the `scikit-learn` library.
- `data.csv`: The Breast Cancer Wisconsin dataset used for classification tasks.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). This dataset includes features related to breast cancer measurements and is used to train and test the classifiers.

## Installation

To run the code in this repository, you'll need to install the necessary libraries. Follow these steps to set up your environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository

### Description

This repository contains two implementations of the Naive Bayes Classifier:

1. **Custom Implementation** (`naive_bayes_classifier_from_scratch.ipynb`): This notebook provides a custom implementation of a Naive Bayes Classifier from scratch. It uses the Gaussian Naive Bayes approach, assuming that the features follow a Gaussian distribution. The notebook covers data loading, preprocessing, model training, and evaluation of the classifier's performance on a binary classification task.

2. **Scikit-Learn Implementation** (`naive_bayes_classifier_sklearn.ipynb`): This notebook demonstrates how to implement a Naive Bayes Classifier using the `scikit-learn` library. It includes steps for loading the dataset, preprocessing, model training, and evaluation using `scikit-learn`'s GaussianNB class.

### Steps

1. **Load the dataset**: Read the dataset and preprocess it.
2. **Preprocessing**: Handle missing values, encode categorical features, and split the data.
3. **Model Training**: Train a Naive Bayes Classifier using the custom implementation.
4. **Evaluation**: Evaluate the classifier's performance using metrics like accuracy, precision, and recall.

### Usage

To run the custom Naive Bayes Classifier notebook, open it in Jupyter Notebook or JupyterLab and run the cells sequentially. You can launch Jupyter Notebook by using:

```bash
jupyter notebook naive_bayes_classifier_from_scratch.ipynb
jupyter notebook naive_bayes_classifier_sklearn.ipynb
