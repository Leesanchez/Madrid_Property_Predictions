import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_excel("data/Session 7 Dataset.xlsx")

# Drop unnecessary index column if exists
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# Handle missing values
for col in df.select_dtypes(include=["float64", "int64"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical variables
categorical_cols = ["inm_barrio", "inm_distrito"]
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Define classification target (e.g., classifying properties as expensive or cheap)
df["price_category"] = pd.qcut(df["inm_price"], q=2, labels=[0, 1])

# Select features and target
X = df.drop(columns=["inm_price", "price_category"])
y = df["price_category"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classification models
models = {
    "Perceptron": Perceptron(),
    "Logistic Regression": LogisticRegression(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Save results to CSV
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
results_df.to_csv("results/classification_results.csv", index=False)

# Save trained models
for name, model in models.items():
    joblib.dump(model, f"models/classification_model_{name}.pkl")