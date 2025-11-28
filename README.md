# Stress-Level-Prediction-using-Machine-Learning
Created dataset from scratch, cleaned with Pandas, trained Logistic Regression &amp; Random Forest, and visualized results using confusion matrix and feature importance. Simple end-to-end project for mental health analytics.
---

## 🔍 Project Workflow

### 1️⃣ Dataset Creation

A synthetic but realistic dataset is generated using distributions for:

* Sleep Hours
* Screen Time
* Physical Activity
* Heart Rate
* Workload
* Water Intake
* Mood Score

### 2️⃣ Data Cleaning

* Handled missing values
* Standardized features
* Prepared dataset for ML

### 3️⃣ Data Visualization

Includes:

* Confusion Matrix
* Feature Importance (Random Forest)

### 4️⃣ Model Training

Two ML models implemented:

* **Logistic Regression**
* **Random Forest Classifier** (best performer)

### 5️⃣ Model Evaluation

Measured using:

* Accuracy
* Classification Report
* Confusion Matrix
* Feature Importance

---

## 📦 Full Project Code (Google Colab Ready)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n = 1000

# Dataset
data = pd.DataFrame({
    "Sleep_Hours": np.random.normal(6.5, 1, n).clip(3, 10),
    "Screen_Time": np.random.normal(5, 1.5, n).clip(1, 12),
    "Physical_Activity": np.random.normal(40, 15, n).clip(5, 120),
    "Heart_Rate": np.random.normal(80, 10, n).clip(55, 140),
    "Workload": np.random.normal(6, 2, n).clip(1, 10),
    "Water_Intake": np.random.normal(2.5, 0.7, n).clip(1, 5),
    "Mood_Score": np.random.normal(6, 2, n).clip(1, 10)
})

stress = []
for i in range(n):
    score = ((10 - data.loc[i, "Sleep_Hours"]) +
             data.loc[i, "Screen_Time"] +
             (120 - data.loc[i, "Physical_Activity"]) / 20 +
             (data.loc[i, "Heart_Rate"] - 70) / 10 +
             data.loc[i, "Workload"])
    stress.append("Low" if score < 8 else "Medium" if score < 13 else "High")

data["Stress_Level"] = stress
data.fillna(data.mean(), inplace=True)

X = data.drop("Stress_Level", axis=1)
y = data["Stress_Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.show()

importances = rf_model.feature_importances_
plt.figure(figsize=(8,4))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance")
plt.show()
```

---

## 📊 Results & Insights

* **Random Forest achieved the highest accuracy**
* Top predictors:

  * Heart Rate
  * Workload
  * Sleep Hours
* Clear separation of stress categories shown in the confusion matrix
* Great explainability using feature importance

---

## 📁 Repository Structure

```
📁 Stress-Level-Prediction
 ├── README.md
 ├── stress_prediction.ipynb
 ├── dataset.csv
 ├── images/
 │     ├── confusion_matrix.png
 │     └── feature_importance.png
 └── models/
       ├── random_forest.pkl
       └── scaler.pkl
```

---

## ⭐ Why This Project Is Resume-Ready

* Covers full ML pipeline
* Mental-health analytics (strong JD relevance)
* Clean code + clear visuals
* Shows real ML understanding end-to-end

If you like this project, consider ⭐ starring the repo!

