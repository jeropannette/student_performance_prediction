import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (make sure it's in the same folder)
data = pd.read_csv("student-mat.csv", sep=';')

# Encode categorical features so that numeric-only plots work
from sklearn.preprocessing import LabelEncoder
categorical_cols = data.select_dtypes(include=['object']).columns

label_enc = LabelEncoder()
for col in categorical_cols:
    data[col] = label_enc.fit_transform(data[col])

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Histogram of Final Grade (G3)
plt.figure(figsize=(8, 5))
sns.histplot(data['G3'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_g3_histogram.png")
plt.show()

# 2. Heatmap of Correlation Matrix
plt.figure(figsize=(15, 10))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_correlation_matrix.png")
plt.show()

# 3. Boxplot: Study Time vs G3
plt.figure(figsize=(8, 5))
sns.boxplot(x='studytime', y='G3', data=data, palette="Set2")
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time (1 = low, 4 = high)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("eda_studytime_boxplot.png")
plt.show()

# 4. Scatterplot: Absences vs Final Grade
plt.figure(figsize=(8, 5))
sns.scatterplot(x='absences', y='G3', data=data)
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("eda_absences_scatter.png")
plt.show()
