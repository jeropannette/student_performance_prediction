import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("student-mat.csv", sep=';')
print("Initial Data Shape:", data.shape)

#  Check for and handle missing values
print("Missing Values per Column:\n", data.isnull().sum())

#  Remove duplicates
initial_rows = data.shape[0]
data.drop_duplicates(inplace=True)
print(f"Removed {initial_rows - data.shape[0]} duplicate rows")

#  Encode categorical variables
label_enc = LabelEncoder()
categorical_columns = data.select_dtypes(include=['object']).columns

for col in categorical_columns:
    data[col] = label_enc.fit_transform(data[col])

#  Handle outliers in final grade
data = data[(data['G3'] >= 0) & (data['G3'] <= 20)]

#  Normalize numerical features (excluding target variable)
X = data.drop('G3', axis=1)
y = data['G3']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
