# Loan Approval Prediction — Logistic Regression vs Random Forest

Machine Learning course project (DS2401), Umm Al-Qura University, June 2026. Team of four: Hamed Fardous, Abdulkarim Mutwalli, Saleh Obaidli and Mohammed Alghamdi.

Binary classification: predict whether a loan application is approved (Y) or rejected (N) from applicant data such as income, credit history, marital status and property area.

## Data
[Kaggle Loan Prediction dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset): 614 records, 13 columns, 68.7% approved / 31.3% rejected. The data file is not included; download it from Kaggle.

## Pipeline
1. **Missing values:** mode for categorical columns, median for `LoanAmount` and `Loan_Amount_Term`.
2. **Encoding:** label encoding for binary columns, one-hot encoding for `Dependents` and `Property_Area`.
3. **Split:** 80% training (491) / 20% test (123).
4. **Models:** Logistic Regression (baseline) and Random Forest (scikit-learn, `random_state=42`).

## Results (123 test records)

| Model | Accuracy | False negatives | False positives |
|---|---|---|---|
| Logistic Regression | 78.86% | 1 | 25 |
| Random Forest | 77.24% | 3 | 25 |

Logistic Regression was preferred: it rejected only one legitimate applicant. Next steps are feature scaling, class-imbalance handling (SMOTE or class weights) and hyperparameter tuning.

## Files
- `ML_Loan_Approval_Project.ipynb` — notebook with the full workflow and outputs
- `Report_ML_Loan_Approval.pdf` — project report
