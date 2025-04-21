# Alphabet Soup Charity: Deep Learning Challenge

A binary classification project using TensorFlow and Keras to help Alphabet Soup, a nonprofit foundation, better identify which applicants are most likely to use funding effectively. This challenge leverages deep learning to build, evaluate, and optimize a neural network model trained on historical funding data from over 34,000 organizations.

## 🔍 Project Overview
Alphabet Soup wants to develop a tool that predicts whether applicants for funding will be successful if chosen. Using a structured dataset of metadata on previously funded organizations, the goal was to:

Preprocess and clean the data
Build and evaluate a baseline deep learning model
Apply optimization strategies to improve model performance
Draw conclusions on model effectiveness and data limitations

## Data Preprocessing
Target Variable: IS_SUCCESSFUL – whether the organization used the funds effectively
Feature Variables: All columns except dropped ones
Dropped Columns: EIN, NAME, STATUS, ASK_AMT, and SPECIAL_CONSIDERATIONS due to imbalance or lack of value
Encoding: One-hot encoding via pd.get_dummies()
Scaling: Feature standardization using StandardScaler

## Model Architecture & Optimization
Baseline Model
Hidden Layers: 2 
Activation Function: ReLU
Output Layer: Sigmoid
Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 100
Accuracy: ~72.83%

## 🔁 Optimization Attempts
Two optimization strategies were tested:

Added a third dense layer
Dropped additional columns and reduced threshold for classification binning
Despite these changes, the models returned consistent results. The best-performing model reached:
Accuracy: 73.4%

## Evaluation
Despite removing 1 low-value columns and experimenting with different models, changing the binning number or nodes the accuracy percentage didnt change very much. This suggests that while the dropped columns added some noise, they weren’t the primary limiting factor.

The main challenge appears to be the dataset itself. Subtle, inconsistent, or imbalanced patterns may be preventing the model from learning effectively. Significant improvement will likely depend on deeper data analysis, refined feature engineering, or higher-quality input data.


