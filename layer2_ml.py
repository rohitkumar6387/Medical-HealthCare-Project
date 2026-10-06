import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
num_patients = 1000

ages = np.random.randint(1, 90, size = num_patients)
fever = np.random.uniform(98.0, 104.5, size = num_patients).round(1)
oxygen_levels = np.random.randint(85, 100, size = num_patients)
cough_severity = np.random.randint(1, 6, size = num_patients)
shortness_of_breath = np.random.choice([0, 1], size = num_patients, p=[0.6, 0.4])

risk_scores = []
for i in range(num_patients):
    score = 0
    if oxygen_levels[i] < 92 or (shortness_of_breath[i] == 1 and fever[i] > 102):
        score = 2
    elif 92 <= oxygen_levels[i] < 95:
        score = 1
    elif ages[i] > 65 or ages[i] < 5:
        if fever[i] > 101:
            score = 1
    risk_scores.append(score)

df = pd.DataFrame({
    "Age":ages,
    "Fever":fever,
    "Oxygen_Level":oxygen_levels,
    "Cough_Severity":cough_severity,
    "Shortness_of_Breath":shortness_of_breath,
    "Risk_Score":risk_scores
})

df.to_csv("patient_mock_data.csv"),

X = df.drop("Risk_Score",axis = 1)
y = df["Risk_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)

ml_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
ml_model = ml_model.fit(X_train, y_train)

accuracy = ml_model.score(X_test, y_test)*100
print(f"ML Model Accuracy:{accuracy:.2f}")

joblib.dump(ml_model, "patient_risk_model.pkl")
print("Model Saved")
