import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -------------------
# Load dataset
# -------------------
data = pd.read_excel('dataset.xlsx',nrows=100)

# Save target variable before dropping
y = data['Category'].copy()

# Drop target + irrelevant columns for features
drop_columns = [
    'Category', 'LocationAbbr', 'LocationDesc', 'DataSource',
    'Data_Value_Unit', 'Data_Value_Type', 'Data_Value_Footnote_Symbol',
    'Data_Value_Footnote', 'Numerator', 'LocationID', 'DataValueTypeID',
    'GeoLocation', 'Geographic Level', 'StateAbbreviation'
]
X = data.drop(columns=drop_columns, errors='ignore')

# Categorical columns to encode
categorical_cols = [
    "Topic", "Question", "Response", "Age", "Sex", "RaceEthnicity",
    "RiskFactor", "RiskFactorResponse", "TopicID", "CategoryID",
    "QuestionID", "ResponseID", "AgeID", "SexID", "RaceEthnicityID",
    "RiskFactorID", "RiskFactorResponseID", "YearStart", "YearEnd"
]

# Fill missing values in categorical columns
X[categorical_cols] = X[categorical_cols].fillna("Unknown")

# Encode categorical columns
for col in categorical_cols:
    if col in X.columns:
        X[col] = LabelEncoder().fit_transform(X[col])

# Drop any remaining NaNs and align target
X = X.dropna()
y = y.loc[X.index]

# Standardize data (required before PCA)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

print(f"Original dataset size: {len(data)}")
print(f"Features after preprocessing: {X_std.shape[1]}")
print(f"Number of classes: {y.nunique()}")
print(f"Class distribution:\n{y.value_counts()}\n")

# -------------------
# PCA with 99% Variance Retention
# -------------------
print("="*70)
print("APPLYING PCA WITH 95% EXPLAINED VARIANCE")
print("="*70)

# First, fit PCA with all components to analyze variance
pca_full = PCA()
pca_full.fit(X_std)

# Calculate cumulative sum of explained variance
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components for 99% variance
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1

print(f"\nOriginal number of features: {X_std.shape[1]}")
print(f"Components needed for 95% variance: {n_components_95}")
print(f"Dimensionality reduction: {X_std.shape[1]} → {n_components_95}")
print(f"Reduction ratio: {(1 - n_components_95/X_std.shape[1])*100:.2f}%")

# Apply PCA with optimal number of components
pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_std)

print(f"\nExplained variance by first 10 components:")
for i in range(min(10, n_components_95)):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} ({cumsum_variance[i]:.4f} cumulative)")

print(f"\nTotal explained variance with {n_components_95} components: {cumsum_variance[n_components_95-1]:.4f}")

# Visualize explained variance
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         pca_full.explained_variance_ratio_, 'bo-', linewidth=2, markersize=4)
plt.axvline(x=n_components_95, color='r', linestyle='--', label=f'{n_components_95} components (95% var)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - Individual Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'ro-', linewidth=2, markersize=4)
plt.axhline(y=0.99, color='g', linestyle='--', label='95% threshold')
plt.axvline(x=n_components_95, color='r', linestyle='--', label=f'{n_components_95} components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------
# Remove outliers using Isolation Forest
# -------------------
print("\n" + "="*70)
print("OUTLIER REMOVAL")
print("="*70)

isolation_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = isolation_forest.fit_predict(X_pca)
X_no_outliers = X_pca[outliers != -1]
y_no_outliers = y[outliers != -1]

print(f"Samples before outlier removal: {len(X_pca)}")
print(f"Samples after outlier removal: {len(X_no_outliers)}")
print(f"Outliers removed: {len(X_pca) - len(X_no_outliers)} ({((len(X_pca) - len(X_no_outliers))/len(X_pca)*100):.2f}%)")

# -------------------
# Split data
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_no_outliers, y_no_outliers, test_size=0.2, random_state=42, stratify=y_no_outliers
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}\n")

# -------------------
# Define classification models
# -------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# -------------------
# Train and evaluate models
# -------------------
print("="*70)
print("MODEL TRAINING AND EVALUATION")
print("="*70)

results = {}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training {name}...")
    print(f"{'='*70}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

# -------------------
# Compare models
# -------------------
print(f"\n{'='*70}")
print("MODEL COMPARISON (WITH PCA - 95% VARIANCE)")
print(f"{'='*70}")

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'CV Mean': [results[m]['cv_mean'] for m in results.keys()],
    'CV Std': [results[m]['cv_std'] for m in results.keys()]
})
comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
print(comparison_df.to_string(index=False))

# -------------------
# Visualizations
# -------------------

# 1. Model comparison bar plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(comparison_df)))
bars1 = plt.barh(comparison_df['Model'], comparison_df['Test Accuracy'], color=colors)
plt.xlabel('Test Accuracy', fontsize=12)
plt.title('Model Comparison - Test Accuracy\n(PCA: 95% Variance)', fontsize=14, fontweight='bold')
plt.xlim([0, 1])
for i, (bar, acc) in enumerate(zip(bars1, comparison_df['Test Accuracy'])):
    plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{acc:.4f}', va='center', fontsize=10)

plt.subplot(1, 2, 2)
bars2 = plt.barh(comparison_df['Model'], comparison_df['CV Mean'], 
                  xerr=comparison_df['CV Std'], color=colors, capsize=5)
plt.xlabel('Cross-Validation Mean Score', fontsize=12)
plt.title('Model Comparison - CV Score\n(PCA: 95% Variance)', fontsize=14, fontweight='bold')
plt.xlim([0, 1])
for i, (bar, cv) in enumerate(zip(bars2, comparison_df['CV Mean'])):
    plt.text(cv + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{cv:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison_pca95.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Confusion matrix for best model
best_model_name = comparison_df.iloc[0]['Model']
best_predictions = results[best_model_name]['predictions']

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title(f'Confusion Matrix - {best_model_name}\n(PCA: 95% Variance)', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_best_model_pca95.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------
# Final Summary
# -------------------
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
print(f"Original Features: {X_std.shape[1]}")
print(f"PCA Components (95% variance): {n_components_95}")
print(f"Dimensionality Reduction: {(1 - n_components_95/X_std.shape[1])*100:.2f}%")
print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {comparison_df.iloc[0]['Test Accuracy']:.4f}")
print(f"Best CV Score: {comparison_df.iloc[0]['CV Mean']:.4f} (+/- {comparison_df.iloc[0]['CV Std']:.4f})")
print(f"{'='*70}")

# Save results to CSV for comparison
comparison_df.to_csv('model_results_pca95.csv', index=False)
print("\nResults saved to 'model_results_pca95.csv'")