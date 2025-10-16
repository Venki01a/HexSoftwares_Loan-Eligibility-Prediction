# 💰 Loan Eligibility Prediction System using Machine Learning 🤖

### 🏢 Built during my internship as a **Data Science Intern at HexSoftwares Company**

---

## 📘 Project Overview

Loan approval plays a crucial role in the financial industry 🏦. Every day, banks receive numerous loan applications — and manually checking each one can be time-consuming and error-prone.  

This project leverages **Machine Learning** to automate the loan eligibility process.  
It predicts whether an applicant’s loan will be **Approved ✅** or **Rejected ❌**, based on key financial and personal details such as income, education, employment, and CIBIL score.  

The goal is to create a **data-driven, fair, and efficient loan evaluation system** that can assist banks and financial institutions in making faster, unbiased, and accurate decisions.  

---

## 📊 Dataset Description

The dataset contains multiple financial and demographic features that influence loan decisions.  

| Feature | Description |
|----------|--------------|
| `no_of_dependents` | Number of dependents of the applicant 👨‍👩‍👧‍👦 |
| `education` | Education level (Graduate / Not Graduate) 🎓 |
| `self_employed` | Employment type (Yes / No) 💼 |
| `income_annum` | Annual income of the applicant 💰 |
| `loan_amount` | Requested loan amount 🏠 |
| `loan_term` | Duration of the loan (in months) ⏱️ |
| `cibil_score` | Applicant’s credit score 💳 |
| `residential_assets_value` | Value of residential assets 🏡 |
| `commercial_assets_value` | Value of commercial assets 🏢 |
| `luxury_assets_value` | Value of luxury assets (e.g., car, jewelry) 💎 |
| `bank_asset_value` | Total bank asset value 🏦 |
| `loan_status` | Target variable – Approved (1) or Rejected (0) ✅❌ |

---

## ⚙️ Machine Learning Models Used

To ensure high accuracy and generalization, we trained and compared multiple models:

1. 🔹 **Logistic Regression** – Simple, interpretable baseline  
2. 🌲 **Random Forest Classifier** – Powerful ensemble method  
3. 🌿 **Decision Tree Classifier** – For easy interpretability  
4. ⚡ **XGBoost Classifier** – Advanced gradient boosting model with top accuracy  

---

## 🧠 Steps Followed

1. **Data Preprocessing**
   - Removed unnecessary columns (`loan_id`)
   - Encoded categorical variables like education and self_employed
   - Scaled continuous variables using `StandardScaler`
   - Handled missing or inconsistent values

2. **Exploratory Data Analysis (EDA)**
   - Visualized feature distributions using Seaborn & Matplotlib
   - Explored correlations between CIBIL score, income, and loan approval
   - Detected outliers and data patterns

3. **Model Training & Evaluation**
   - Split dataset into training and testing sets (80-20)
   - Trained multiple ML models
   - Compared metrics: Accuracy, Precision, Recall, and F1-score

4. **Model Selection**
   - Selected **XGBoost Classifier** for deployment with the best performance (98% accuracy)
   - Saved trained model as `loan_eligibility_model.pkl` using `pickle`

---

## 📈 Model Performance Comparison

| Model | Accuracy (%) | Precision | Recall | F1-Score |
|--------|--------------|-----------|--------|-----------|
| Logistic Regression | 91.2 | 0.90 | 0.88 | 0.89 |
| Decision Tree | 95.4 | 0.95 | 0.94 | 0.94 |
| Random Forest | 97.1 | 0.97 | 0.96 | 0.96 |
| **XGBoost Classifier** | **98.0** ✅ | **0.98** | **0.97** | **0.98** |

> 🏆 The **XGBoost Classifier** outperformed all other models with the highest accuracy, making it ideal for real-world deployment.

---

## 🧩 Technologies Used

| Category | Tools/Technologies |
|-----------|--------------------|
| Programming Language | Python 🐍 |
| Libraries | Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn |
| Environment | Jupyter Notebook (.ipynb) |
| Model Saving | Pickle |

---

## 🌍 Real-World Applications

- 🏦 **Banking Systems** – Automating the loan approval process  
- 💻 **FinTech Platforms** – Real-time credit scoring and eligibility checks  
- 📊 **Business Analytics** – Identifying financial risk patterns  
- ⚙️ **AI-driven Decision Support** – Making fair and quick loan decisions  

---

## 🎯 Key Insights

- Financial behavior patterns like **CIBIL score**, **income**, and **loan amount** are the strongest predictors of loan approval.  
- Machine Learning models can drastically reduce the manual verification time.  
- Using ensemble models like **Random Forest** and **XGBoost** ensures high accuracy and low overfitting.  
- This project demonstrates how **AI can make banking smarter, faster, and fairer**.  

---


