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

X = df[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
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

# Ensure clean ascending order so largest values appear at the top of the plot
plot_data = odds_ratios.sort_values(by='Odds_Ratio', ascending=True)

plt.figure(figsize=(9, 5))

# Plot horizontal bars
bars = plt.barh(plot_data['Feature'], plot_data['Odds_Ratio'], color='steelblue', edgecolor='black')

# Color bars conditionally: green for OR > 1, red for OR < 1
for bar, or_val in zip(bars, plot_data['Odds_Ratio']):
    bar.set_color('#2b8cbe' if or_val >= 1.0 else '#e34a33')

# Add the reference line where OR = 1 (no effect)
plt.axvline(x=1.0, color='black', linestyle='--', linewidth=1.2, label='No Effect (OR = 1.0)')

# Annotate exact numbers on each bar
for index, row in plot_data.reset_index().iterrows():
    plt.text(row['Odds_Ratio'] + 0.05, index, f"{row['Odds_Ratio']:.2f}", 
             va='center', fontsize=10, fontweight='bold')

plt.title('Impact on Titanic Survival Odds (Per 1 Std Dev)', fontsize=13, pad=15)
plt.xlabel('Odds Ratio (exp(coef))', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.xlim(0, max(plot_data['Odds_Ratio']) * 1.15)
plt.legend(loc='lower right')
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
