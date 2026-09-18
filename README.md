# Stroke Prediction Using Machine Learning

## Student Details

- **Name:** Janani B
- **Registration Number:** 25MIM10094
- **Program:** Integrated M.Tech AI Branch
- **Course:** Fundamentals of Artificial Intelligence and Machine Learning
- **University:** VIT Bhopal University

---

## 1. Project Overview

This project implements a machine-learning-based Stroke Prediction System.

The system uses patient-related health and demographic information to predict whether a patient belongs to the stroke-risk class.

The project demonstrates a complete AI/ML workflow:

**Dataset → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → Prediction**

The machine-learning model used in this project is **Logistic Regression**, a supervised learning algorithm suitable for binary classification.

> **Disclaimer:** This project is developed for academic and educational purposes. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 2. Problem Statement

Stroke is a serious medical condition where early identification of risk factors can support preventive decision-making.

The objective of this project is to develop a machine-learning classification system that learns patterns from historical patient data and predicts the possibility of stroke based on available patient attributes.

The prediction problem is treated as a **binary classification problem**:

- `0` → No stroke
- `1` → Stroke

---

## 3. Objectives

The main objectives of this project are:

1. Load and preprocess a stroke dataset.
2. Handle missing and categorical data.
3. Convert categorical features into machine-readable numerical representations.
4. Train a supervised machine-learning model.
5. Evaluate the model using classification metrics.
6. Generate stroke predictions for new patient data.
7. Demonstrate an AI/ML workflow that can be executed from the command line.

---

## 4. Dataset

The project uses a stroke prediction dataset containing patient demographic and health-related attributes.

Important features include:

| Feature | Description |
|---|---|
| `gender` | Patient gender |
| `age` | Patient age |
| `hypertension` | Hypertension indicator |
| `heart_disease` | Heart disease indicator |
| `ever_married` | Marriage status |
| `work_type` | Type of employment |
| `Residence_type` | Urban or rural residence |
| `avg_glucose_level` | Average glucose level |
| `bmi` | Body Mass Index |
| `smoking_status` | Smoking status |
| `stroke` | Target variable |

The target variable is:

```text
stroke
