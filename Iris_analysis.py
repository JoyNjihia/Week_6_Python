# iris_analysis.py

"""
Author: [Joyce Nyambura Njihia]
Project: Iris Dataset Analysis
Description: This script loads and analyzes the Iris dataset, performs basic statistics,
visualizes the data, and includes error handling and insights.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# Task 1: Load and Explore the Dataset


try:
    # Load Iris dataset
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Clean dataset: Drop missing values (none in this dataset)
df.dropna(inplace=True)


# Task 2: Basic Data Analysis


# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Group by species and calculate mean
grouped = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped)

# Observation
print("\nObservation:")
print("- Setosa has the smallest petal size.")
print("- Virginica has the largest petal size.")
print("- Versicolor is in between.")


# Task 3: Data Visualization


sns.set(style="whitegrid")

# Line Chart: Sepal Length trend over index (simulated time)
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['sepal length (cm)'], color='green', label='Sepal Length')
plt.title('Sepal Length Trend Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Chart: Average Petal Length per Species
plt.figure(figsize=(6, 4))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'], palette='pastel')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# Histogram: Sepal Width Distribution
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set2')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()
