# Prediction of Student Placement Using Machine Learning

This project focuses on predicting whether a student will be placed or not using machine learning models. The dataset (`placement.csv`) consists of various academic and non-academic features related to student performance. A pre-trained model (`model.pkl`) is used for predictions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Student placement prediction is crucial for academic institutions and students. This project aims to automate the prediction process using machine learning techniques applied to student performance data.

## Dataset

- **File:** `placement.csv`
- **Format:** CSV
- **Features:** Academic scores, extracurricular activities, projects, internships, etc.
- **Target Variable:** Placed (Yes/No)
- **Preprocessing Required:** Data cleaning, feature selection, and encoding

## Preprocessing

1. Load student data from `placement.csv`.
2. Handle missing values and encode categorical variables.
3. Feature scaling and selection.
4. Split the dataset into training and testing sets.

## Model Training

- Implemented machine learning models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- The final trained model is stored as `model.pkl`.
- Training is performed using **scikit-learn**.

Run the training script:

```bash
python train.py --model random_forest --epochs 50
```

## Evaluation

The models are evaluated based on:

- Accuracy
- Precision, Recall

Run evaluation script:

```bash
python evaluate.py --model random_forest
```

## Usage

To test the trained model (`model.pkl`) on new student data:

```bash
python predict.py --input new_student_data.csv --model model.pkl
```

## Results

- Achieved **80% accuracy** on student placement dataset.
- Feature engineering played a significant role in model performance
