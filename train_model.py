import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("loan_approval_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("\n========== ORIGINAL DATASET ==========")
print(df.shape)


# ==============================
# 2. REMOVE LOAN ID
# ==============================

df = df.drop("loan_id", axis=1)

print("\n========== AFTER REMOVING LOAN ID ==========")
print(df.shape)


# ==============================
# 3. CHECK TARGET DISTRIBUTION
# ==============================

print("\n========== LOAN STATUS ==========")
print(df["loan_status"].value_counts())


# ==============================
# 4. CONVERT CATEGORICAL VALUES
# ==============================

df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()


# Convert categorical columns to numbers

df["education"] = df["education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})


# ==============================
# 5. CHECK DATA AFTER ENCODING
# ==============================

print("\n========== DATA AFTER ENCODING ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES AFTER ENCODING ==========")
print(df.isnull().sum())


# ==============================
# 6. EXPLORATORY DATA ANALYSIS
# ==============================

sns.set_theme(style="whitegrid")


# Loan Status Distribution

plt.figure(figsize=(7, 5))

sns.countplot(
    x=df["loan_status"],
    hue=df["loan_status"],
    legend=False
)

plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status (0 = Rejected, 1 = Approved)")
plt.ylabel("Number of Applications")

plt.show()


# CIBIL Score vs Loan Status

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["loan_status"],
    y=df["cibil_score"]
)

plt.title("CIBIL Score vs Loan Status")
plt.xlabel("Loan Status (0 = Rejected, 1 = Approved)")
plt.ylabel("CIBIL Score")

plt.show()


# Income vs Loan Status

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["loan_status"],
    y=df["income_annum"]
)

plt.title("Annual Income vs Loan Status")
plt.xlabel("Loan Status (0 = Rejected, 1 = Approved)")
plt.ylabel("Annual Income")

plt.show()


# Loan Amount vs Loan Status

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["loan_status"],
    y=df["loan_amount"]
)

plt.title("Loan Amount vs Loan Status")
plt.xlabel("Loan Status (0 = Rejected, 1 = Approved)")
plt.ylabel("Loan Amount")

plt.show()


# Correlation Heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.show()