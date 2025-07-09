# 📉 Customer Churn Prediction - Day 4 | 100 Days of Data Science

This project focuses on predicting **telecom customer churn** using supervised machine learning models. It is part of my `#100DaysOfDataScience` challenge.

## 📦 Dataset

We used the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which includes customer-level data on:

- Demographics (Gender, SeniorCitizen, etc.)
- Services signed up for (Internet, Phone, Streaming, etc.)
- Account information (Tenure, Charges, Contract type)
- Churn status (Target)

**Shape:** `7043 rows × 21 columns`  
**Target Variable:** `Churn` (Yes/No → 1/0)

---

## 🚀 Objective

To build a model that predicts whether a customer will churn or not, and identify key factors that influence churn.

---

## 🔧 Steps Followed

### ✅ 1. Data Preprocessing
- Removed customerID column
- Checked for nulls and cleaned whitespace entries in `TotalCharges`
- Encoded categorical variables using `LabelEncoder`
- Converted target (`Churn`) into binary

### 📊 2. EDA (Exploratory Data Analysis)
- Checked churn rate
- Analyzed numerical distributions (Tenure, MonthlyCharges)
- Found class imbalance (approx. 26.5% churn rate)

### 🧠 3. Model Building
- Train-test split (80-20)
- Feature Scaling with `StandardScaler`
- Models used:
  - Logistic Regression
  - Random Forest Classifier

### ⚖️ 4. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance churn labels

---

## 📈 Model Results

| Model                | Accuracy | Precision (Churn) | Recall (Churn) | F1-score (Churn) |
|---------------------|----------|-------------------|----------------|------------------|
| Logistic Regression | 81.7%    | 0.69              | 0.57           | 0.62             |
| Random Forest        | 79.9%    | 0.67              | 0.49           | 0.56             |
| **LogReg + SMOTE**   | 75.5%    | 0.52              | 0.81           | 0.64             |
| **RF + SMOTE**       | 78.8%    | 0.60              | 0.59           | 0.60             |

---

## 📊 Visualizations

- 📌 Confusion Matrix Heatmaps
- 📌 Actual vs Predicted Distribution Plot
- 📌 🔥 Feature Importance (from Random Forest)

### Top features influencing churn:
- Contract Type
- Tenure
- Monthly Charges
- Total Charges

---

## 💾 Model Saving

Final model (Random Forest with SMOTE) was saved using:
```python
import pickle

with open('churn_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)



📌 Learnings & Reflections
📉 High accuracy doesn't always mean good churn prediction — class imbalance can mislead.

🧪 SMOTE helped improve recall for churn class, showing how resampling techniques matter.

🔍 Feature importance provided business insights: short tenure and month-to-month contracts are strong churn signals.

💡 Saving models and preparing them for deployment is an essential step toward production-ready projects.

