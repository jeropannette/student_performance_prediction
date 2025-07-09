import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("student-mat.csv", sep=';')

# Feature engineering
data['avg_grade'] = (data['G1'] + data['G2']) / 2

# Encode categorical columns
categorical_cols = data.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Drop G1 and G2
data.drop(['G1', 'G2'], axis=1, inplace=True)

# Define only the selected 7 features
selected_features = ['avg_grade', 'failures', 'studytime', 'absences', 'goout', 'freetime', 'internet']
X = data[selected_features]
y = data['G3']

# ✅ Fit the scaler ONLY on selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the correct scaler
joblib.dump(scaler, "scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "rf_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model and Scaler saved successfully!")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
