import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset 
data = pd.read_csv("student-mat.csv", sep=';')


 #Combine Previous Grades to Create an Average Grade Feature

data['avg_grade'] = (data['G1'] + data['G2']) / 2  # new feature


# Binary Feature – High Absenteeism

data['high_absentee'] = data['absences'].apply(lambda x: 1 if x > 10 else 0)

-
# Encode Categorical Columns

categorical_cols = data.select_dtypes(include='object').columns
label_enc = LabelEncoder()

for col in categorical_cols:
    data[col] = label_enc.fit_transform(data[col])

data.drop(['G1', 'G2'], axis=1, inplace=True)

# -------------------------------
# Step 5: Scale Numerical Features
# -------------------------------
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop('G3')  # exclude target

scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# -------------------------------
# Final Split for Modeling
# -------------------------------
X = data.drop('G3', axis=1)
y = data['G3']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Feature Engineering Complete.")
print("Final Feature Set Shape:", X_train.shape)
