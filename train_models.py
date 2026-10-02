import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("loan_approval_dataset.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove spaces from text values
df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()


# ==========================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================

df = df.drop("loan_id", axis=1)


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("loan_status", axis=1)

y = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})


# ==========================================
# 4. IDENTIFY COLUMNS
# ==========================================

categorical_columns = [
    "education",
    "self_employed"
]

numerical_columns = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========================================")
print("TRAINING AND TESTING DATA")
print("========================================")

print("Total records :", len(X))
print("Training data :", len(X_train))
print("Testing data  :", len(X_test))


# ==========================================
# 6. PREPROCESSING
# ==========================================

preprocessor_scaled = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


preprocessor_tree = ColumnTransformer(
    transformers=[
        (
            "numerical",
            "passthrough",
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# ==========================================
# 7. CREATE MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("preprocessing", preprocessor_scaled),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessing", preprocessor_tree),
        ("model", DecisionTreeClassifier(
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessing", preprocessor_tree),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("preprocessing", preprocessor_scaled),
        ("model", SVC())
    ])
}


# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================

results = {}

best_model = None
best_model_name = None
best_f1 = -1


for name, model in models.items():

    print("\n")
    print("========================================")
    print(name)
    print("========================================")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Rejected", "Approved"]
        )
    )

    # Select best model using F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# ==========================================
# 9. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results).T

print("\n\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(results_df)


# ==========================================
# 10. BEST MODEL
# ==========================================

print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Best Model:", best_model_name)
print(f"Best F1-Score: {best_f1:.4f}")


# ==========================================
# 11. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "loan_approval_model.pkl"
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print("File: loan_approval_model.pkl")