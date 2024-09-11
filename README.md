# Naive Bayes Classifier Implementation

## Overview

This repository contains implementations of a Naive Bayes Classifier for binary classification tasks using both a custom implementation and the `scikit-learn` library. The focus is on the classification of breast cancer data from the Breast Cancer Wisconsin dataset.

## Contents

- `naive_bayes_classifier_from_scratch.ipynb`: Custom implementation of Naive Bayes Classifier from scratch without using libraries.
- `naive_bayes_classifier_sklearn.ipynb`: Implementation of Naive Bayes Classifier using the `scikit-learn` library.
- `data.csv`: The Breast Cancer Wisconsin dataset used for classification tasks.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). This dataset includes features related to breast cancer measurements and is used to train and test the classifiers.

## Custom Naive Bayes Classifier (`naive_bayes_classifier.py`)

### Description

This script provides a custom implementation of a Naive Bayes Classifier from scratch. The classifier is implemented using a Gaussian Naive Bayes approach, assuming that the features follow a Gaussian distribution.

### Steps

1. **Load the dataset**: Read the dataset and preprocess it.
2. **Preprocessing**: Handle missing values, encode categorical features, and split the data.
3. **Model Training**: Train a Naive Bayes Classifier using the custom implementation.
4. **Evaluation**: Evaluate the classifier's performance using metrics like accuracy, precision, and recall.

### Usage

```python
python naive_bayes_classifier.py
