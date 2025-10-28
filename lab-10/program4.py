import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular
import shap
import warnings
warnings.filterwarnings('ignore')


# ----------------------------
# 1. Load and preprocess data
# ----------------------------
def load_data(filepath, target_col='Response'):
    df = pd.read_excel(filepath, nrows=100)
    df = df.dropna(subset=[target_col])
    
    y = df[target_col]
    if y.dtype == object or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)
    
    X = df.select_dtypes(include=np.number).dropna(axis=1, how='all')
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    
    return X, y.ravel()


# ----------------------------
# 2. Sequential Feature Selection
# ----------------------------
def run_sfs(X, y, n_features=5):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    sfs = SequentialFeatureSelector(clf, n_features_to_select=n_features, direction='forward')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_sfs = sfs.fit_transform(X_train, y_train)
    X_test_sfs = sfs.transform(X_test)
    
    clf.fit(X_train_sfs, y_train)
    y_pred = clf.predict(X_test_sfs)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, X_train_sfs, X_test_sfs, y_train, y_test, accuracy, sfs.get_feature_names_out(X.columns)


# ----------------------------
# 3. Model Explainability
# ----------------------------
def explain_model(clf, X_train_sfs, X_test_sfs, feature_names, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sfs)
    X_test_scaled = scaler.transform(X_test_sfs)
    
    # LIME Explainer
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names.tolist(),
        class_names=[str(c) for c in np.unique(y_train)],
        mode='classification'
    )
    instance_idx = 0
    lime_exp = explainer_lime.explain_instance(
        data_row=X_test_scaled[instance_idx],
        predict_fn=clf.predict_proba,
        num_features=len(feature_names)
    )
    
    # SHAP Explainer
    explainer_shap = shap.TreeExplainer(clf)
    shap_values = explainer_shap.shap_values(X_test_scaled)
    
    return lime_exp, explainer_shap, shap_values, X_test_scaled, instance_idx


# ----------------------------
# 4. Main Execution
# ----------------------------
if __name__ == "_main_":
    X, y = load_data('dataset.xlsx', target_col='Response')
    
    print("### Task A4: Sequential Feature Selection ###")
    clf, X_train_sfs, X_test_sfs, y_train, y_test, acc, feature_names = run_sfs(X, y, n_features=5)
    print(f"SFS Accuracy (5 selected features): {acc:.4f}")
    print(f"Selected Features: {feature_names}")
    print("="*70)
    
    print("### Task A5: Explainability with LIME and SHAP ###")
    lime_exp, shap_explainer, shap_vals, X_test_scaled, idx = explain_model(
        clf, X_train_sfs, X_test_sfs, feature_names, y_train, y_test
    )
    
    # Show LIME explanation
    print(f"\n--- LIME Explanation for Test Sample #{idx} ---")
    print(f"True Label: {y_test.iloc[idx]}")
    for feat, weight in lime_exp.as_list():
        print(f"{feat}: {weight:.4f}")
    
    # SHAP summary plot
    print("\nGenerating SHAP Summary Plot...")
    shap.summary_plot(shap_vals, X_test_scaled, feature_names=feature_names, show=False)
    plt.show()