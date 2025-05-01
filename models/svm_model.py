import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")  # Suppress warnings

# ✅ Load Features & Labels
feature_path = "C:/Users/nniha/flask_env/Scripts/project/models/features.npy"
label_path = "C:/Users/nniha/flask_env/Scripts/project/models/labels.npy"

X = np.load(feature_path)
y = np.load(label_path)

print(f"✅ Features Loaded: {X.shape}, Labels Loaded: {y.shape}")
print(f"📊 First 5 Labels: {y[:5]}")  # Check label distribution

# ✅ Handle Class Imbalance using SMOTE
print("\n⚙️ Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"📊 Resampled Data Shape: {X_resampled.shape}, Labels Shape: {y_resampled.shape}")
print(f"🔍 First 5 Resampled Labels: {y_resampled[:5]}")  # Verify balancing

# ✅ Split Data
print("\n✂️ Splitting dataset into train & test...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)
print(f"📊 Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# ✅ SVM Hyperparameter Tuning
print("\n🎯 Training SVM with GridSearchCV...")
svm_params = {
    "C": [1, 10, 100],
    "gamma": [0.01, 0.1, 1],
    "kernel": ["rbf"]
}

svm = GridSearchCV(SVC(probability=True), svm_params, cv=5, scoring="accuracy", n_jobs=-1)
svm.fit(X_train, y_train)
best_svm = svm.best_estimator_

print(f"✅ Best SVM Parameters: {svm.best_params_}")

# ✅ Train Random Forest
print("\n🌲 Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
print("✅ Random Forest Training Complete!")

# ✅ Train XGBoost
print("\n🚀 Training XGBoost Classifier...")
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
xgb.fit(X_train, y_train)
print("✅ XGBoost Training Complete!")

# ✅ Stacking Classifier
print("\n🔗 Training Stacked Ensemble Classifier...")
stacked_model = StackingClassifier(
    estimators=[
        ("svm", best_svm),
        ("rf", rf),
        ("xgb", xgb)
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)
stacked_model.fit(X_train, y_train)
print("✅ Stacked Ensemble Training Complete!")

# ✅ Model Evaluations
def evaluate_model(model, name):
    print(f"\n📊 Evaluating {name}...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {acc:.2%}")
    print(f"📊 Classification Report ({name}):\n{classification_report(y_test, y_pred)}")
    
    return acc

svm_acc = evaluate_model(best_svm, "SVM")
rf_acc = evaluate_model(rf, "Random Forest")
xgb_acc = evaluate_model(xgb, "XGBoost")
ensemble_acc = evaluate_model(stacked_model, "Stacked Ensemble")

# ✅ Save the Best Model
print("\n💾 Saving the best model...")
best_model = stacked_model if ensemble_acc > max(svm_acc, rf_acc, xgb_acc) else max([(svm_acc, best_svm), (rf_acc, rf), (xgb_acc, xgb)], key=lambda x: x[0])[1]
joblib.dump(best_model, "C:/Users/nniha/flask_env/Scripts/project/models/best_model.pkl")
print("✅ Best Model Saved Successfully!")
