"""
Clinical Alert Bypass Prediction
--------------------------------
This module predicts whether a doctor will bypass a clinical decision support alert.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
import shap


def load_and_filter_data(filepath: str) -> pd.DataFrame:
    """Loads the dataset and applies temporal and feature filtering."""
    df = pd.read_csv(filepath)
    
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df[df['TimeStamp'].dt.month.isin([1, 4, 11]) & (df['TimeStamp'].dt.year == 2025)]
    
    # Feature Selection
    features_to_keep = [
        'TimeStamp', 'Department', 'recordType', 'Alert_Age', 'Gender', 'Height', 
        'Weight', 'eGFR', 'LiverALP', 'LiverALT', 'LiverAST', 'Number_Drugs', 
        'Alert_Group_Title', 'Alert_Severity', 'Alert_Category', 'Involved1Type', 
        'Involved2Type', 'Involved3Type', 'Involved4Type', 'BypassGroupType'
    ]
    return df[features_to_keep].copy()


def initial_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Applies basic typecasting, outlier handling, and temporal feature engineering."""
    df = df.drop_duplicates().copy()

    # Typecasting
    numeric_cols = ['Alert_Age', 'Height', 'Weight', 'eGFR', 'LiverALP', 'LiverALT', 'LiverAST', 'Number_Drugs']
    categorical_cols = ['Department', 'recordType', 'Alert_Group_Title', 'Alert_Category', 
                        'Involved1Type', 'Involved2Type', 'Involved3Type', 'Involved4Type']

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[categorical_cols] = df[categorical_cols].fillna("None").astype(str)
    
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0).astype(int)
    
    # Temporal Feature Engineering
    df['is_night_shift'] = np.where((df['TimeStamp'].dt.hour >= 23) | (df['TimeStamp'].dt.hour < 7), 1, 0)
    df['is_weekend'] = (df['TimeStamp'].dt.dayofweek >= 5).astype(int)
    df = df.drop(columns='TimeStamp')
    
    # Outlier handling
    df['Weight'] = df['Weight'].replace(0, np.nan)
    
    return df


def engineer_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits imputations and encodings strictly on the training set, 
    then applies them to both train and test sets to prevent data leakage.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 1. Alert Severity Handling
    for _df in (X_train, X_test):
        _df["Alert_Severity"] = _df["Alert_Severity"].fillna(0)
        _df["Alert_Severity"] = _df["Alert_Severity"].replace({5.0: 4.0, 9.0: 4.0})

    # 2. Group-specific median imputation for Height/Weight
    for _df in (X_train, X_test):
        _df["Age_Group"] = pd.cut(
            _df["Alert_Age"], bins=[-1, 20, 40, 60, 80, 150], labels=["0-20", "21-40", "41-60", "61-80", "80+"]
        )

    cols_to_impute = ["Height", "Weight"]
    group_medians = X_train.groupby(["Gender", "Age_Group"], observed=True)[cols_to_impute].median()
    gender_medians = X_train.groupby("Gender", observed=True)[cols_to_impute].median()
    global_medians = X_train[cols_to_impute].median()

    for col in cols_to_impute:
        for _df in (X_train, X_test):
            # Apply hierarchical fallback medians (if there are no matching groups)
            gm = _df.set_index(["Gender", "Age_Group"]).index.map(group_medians[col])
            _df[col] = _df[col].fillna(pd.Series(gm, index=_df.index))
            _df[col] = _df[col].fillna(_df["Gender"].map(gender_medians[col]))
            _df[col] = _df[col].fillna(global_medians[col])

    X_train.drop(columns=["Age_Group"], inplace=True)
    X_test.drop(columns=["Age_Group"], inplace=True)

    # 3. Lab Features Imputation & Missing Flags
    lab_features = ["eGFR", "LiverALP", "LiverALT", "LiverAST"]
    for _df in (X_train, X_test):
        missing_flags = _df[lab_features].isna().astype(int).add_suffix("_is_missing")
        _df[missing_flags.columns] = missing_flags

    lab_medians = X_train[lab_features].median()
    X_train[lab_features] = X_train[lab_features].fillna(lab_medians)
    X_test[lab_features] = X_test[lab_features].fillna(lab_medians)

    # 4. Department Translation
    translation_map = {
        "כירורגיה אורתופדית": "Orthopedic Surgery",
        "משפחה, פנימית וכללית": "Internal Medicine",
        "נשים - גינקולוגיה": "OBGYN",
        "עיניים": "Ophthalmology",
        "פסיכיאטריה למבוגרים": "Adult Psychiatry",
        "רפואת ילדים": "Pediatrics",
    }
    for _df in (X_train, X_test):
        if "Department" in _df.columns:
            _df["Department"] = _df["Department"].replace(translation_map)

    # 5. Lumping Rare Categories (Fit on Train, Apply to Both Train and Test)
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        counts = X_train[col].value_counts()
        if len(counts) > 10:
            threshold = len(X_train) * 0.01
            rare_categories = counts[counts < threshold].index
            X_train[col] = X_train[col].replace(rare_categories, "Other")
            X_test[col]  = X_test[col].replace(rare_categories, "Other")

    # 6. One Hot Encoding & Alignment
    cols_to_encode = ["Department", "recordType", "Alert_Group_Title", "Alert_Category",
                      "Involved1Type", "Involved2Type", "Involved3Type", "Involved4Type", "Alert_Severity"]
    
    X_train = pd.get_dummies(X_train, columns=cols_to_encode, dtype=int)
    X_test  = pd.get_dummies(X_test, columns=cols_to_encode, dtype=int)

    # Align columns so test has exactly train's feature space
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test


def train_and_evaluate(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """Trains the Random Forest model, evaluates via CV and Holdout, and displays SHAP values."""
    
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)

    # Cross Validation 
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
    
    print(f"CV ROC-AUC Scores (5 folds): {cv_scores}")
    print(f"Average CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})\n")

    # Final Training & Prediction
    rf_model.fit(X_train, y_train)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    y_pred = rf_model.predict(X_test)

    # Evaluation Output
    print(f"Final Test ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', ax=ax1)
    ax1.set_title("Confusion Matrix")
    
    RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax2.set_title("ROC Curve")
    plt.tight_layout()
    plt.show()

    # SHAP Explanability
    print("Generating SHAP Explanations...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    # Safe fallback for different SHAP versions (they return a different shape)
    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1]  
    elif len(np.array(shap_values).shape) == 3:
        shap_values_class_1 = shap_values[:, :, 1]
    else:
        shap_values_class_1 = shap_values

    shap.summary_plot(shap_values_class_1, X_test)


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    
    # 1. Load and clean raw data
    data_path = os.path.join('data', 'bypass_dataset.csv')
    raw_df = load_and_filter_data(data_path)
    clean_df = initial_preprocessing(raw_df)

    # 2. Define Target and Features
    y = (clean_df["BypassGroupType"] == "Doctor").astype(int)
    X = clean_df.drop(columns=["BypassGroupType"])

    # 3. Global Train/Test Split (Before Imputations to prevent leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Feature Engineering (Fit on Train, Transform Both)
    X_train_processed, X_test_processed = engineer_features(X_train_raw, X_test_raw)

    # 5. Train, Evaluate, and Explain
    train_and_evaluate(X_train_processed, X_test_processed, y_train, y_test)