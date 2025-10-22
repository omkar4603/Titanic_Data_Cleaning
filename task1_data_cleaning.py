# Task 1: Data Cleaning and Preprocessing - Titanic Dataset

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("titanic.csv")

# Explore data
print(" First 5 Rows:\n", df.head())
print("\n Dataset Info:\n")
print(df.info())
print("\n Summary Statistics:\n", df.describe())
print("\n Missing Values:\n", df.isnull().sum())

#  Handle missing values safely (no warnings)
df.fillna({
    'Age': df['Age'].median(),
    'Embarked': df['Embarked'].mode()[0]
}, inplace=True)

# Drop unnecessary column
df.drop(columns=['Cabin'], inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Feature scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Detect and remove outliers
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot (Before Outlier Removal)")
plt.show()

#  Safe outlier removal (won’t remove all data)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

before = len(df)
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
after = len(df)
print(f"\nRemoved {before - after} outliers. Remaining rows: {after}")

#  Save cleaned dataset to absolute path
output_path = r"C:\Users\omkar\OneDrive\Desktop\AI_ML Internship\Task1_Data_Cleaning\cleaned_titanic.csv"
df.to_csv(output_path, index=False)

print(f"\n Data cleaning complete! Cleaned file saved at: {output_path}")
