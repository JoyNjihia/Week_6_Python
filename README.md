# Week_6_Python

# Iris Dataset Analysis

This project performs a basic data analysis and visualization of the **Iris dataset**, one of the most well-known datasets in machine learning and statistics.

## Author:
[Joyce Nyambura Njihia]

## Dataset:
The **Iris dataset** is loaded directly from `sklearn.datasets` and contains 150 samples of iris flowers categorized into 3 species:
- Setosa
- Versicolor
- Virginica

Each sample includes 4 numerical features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Project Structure:
- `iris_analysis.py` – Main Python script that includes:
  - Data loading and inspection
  - Basic statistical analysis
  - Group-wise aggregation
  - Data visualizations using Matplotlib and Seaborn
  - Error handling and insights

## Tasks Covered:

### 1. Data Loading & Exploration
- Loaded the dataset using `pandas` and `sklearn`
- Displayed sample data and structure
- Checked data types and missing values
- Cleaned missing data (if any)

### 2. Basic Analysis
- Calculated summary statistics using `.describe()`
- Grouped data by species to compute mean feature values
- Highlighted meaningful patterns and comparisons

### 3. Data Visualization
Included four custom visualizations with titles, axis labels, and legends:
- **Line Chart**: Sepal length over index (simulated trend)
- **Bar Chart**: Average petal length per species
- **Histogram**: Sepal width distribution
- **Scatter Plot**: Sepal length vs. petal length by species

## Libraries Used:
- `pandas`
- `matplotlib`
- `seaborn`
