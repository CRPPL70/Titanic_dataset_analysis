# --- DATASET ---
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("yasserh/titanic-dataset")

# print("Path to dataset files:", path)

# --- IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Helps with drawing correlation matrix

df = pd.read_csv("Titanic-Dataset.csv")

df = df.drop(['Name','Ticket','Cabin', 'Embarked'], axis=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Compute correlation matrix
correlation_matrix = df.corr()

# Drawing a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Working conditions and mental health correlation matrix')
plt.show()
