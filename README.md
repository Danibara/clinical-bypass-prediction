\# Clinical Alert Bypass Prediction Pipeline



\## Project Overview

This project provides a machine learning solution to predict whether a medical doctor will bypass a clinical decision support alert. By identifying alerts with high bypass rates, healthcare systems can refine alert logic, reduce "alert fatigue," and improve clinical safety.



\## Technical Approach

\- \*\*Data Filtering:\*\* Extracting specific temporal windows (Jan, April, Nov 2025).

\- \*\*Leakage Prevention:\*\* Strict separation of training and testing data before any feature engineering or imputation.

\- \*\*Hierarchical Imputation:\*\* Biometric data (Height/Weight) is imputed using a fallback strategy: Age-Group/Gender median -> Gender median -> Global median.

\- \*\*Model:\*\* Random Forest Classifier with balanced class weights to handle behavioral noise and class imbalance.

\- \*\*Explainability:\*\* SHAP (SHapley Additive exPlanations) used to visualize feature impact on clinical decisions.



