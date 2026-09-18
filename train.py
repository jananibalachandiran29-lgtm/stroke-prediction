import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# 1. Load dataset
data = pd.read_csv("stroke-data.csv")

# 2. Remove unnecessary ID column
if "id" in data.columns:
    data = data.drop("id", axis=1)

# 3. Separate features and target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# 4. Identify numerical and categorical columns
numerical_features = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

# 5. Preprocessing for numerical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Preprocessing for categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 8. Create Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# 9. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 10. Train model
model.fit(X_train, y_train)

# 11. Make predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# 12. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("=" * 50)
print("STROKE PREDICTION MODEL TRAINING")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 13. Save the complete trained pipeline
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as model.pkl")
print("=" * 50)
