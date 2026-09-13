# --- DATASET ---
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("yasserh/titanic-dataset")

# print("Path to dataset files:", path)

# --- IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Helps with drawing correlation matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("Titanic-Dataset.csv")

df = df.drop(['Name','Ticket','Cabin', 'Embarked'], axis=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Compute correlation matrix
correlation_matrix = df.corr()

# Drawing a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Titanic survival correlation matrix')
plt.show()

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Using logistic regression to determine survivor traits
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

coefficients = model.coef_.ravel()

# Calculate odds ratios
odds_ratios = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Odds_Ratio': np.exp(coefficients)
}).sort_values(by='Odds_Ratio', ascending=False)

print(odds_ratios)
