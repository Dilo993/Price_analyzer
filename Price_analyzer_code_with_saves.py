"""
Price Analyzer with Saves

This script provides functionality for analyzing the contribution of various features
(columns) in a dataset to a target variable (e.g., price). It uses a Random Forest Regressor
to calculate feature importance and saves the results to output files. Additionally, it
generates visualizations of the average contributions of features.

Key Features:
- Load a CSV file for analysis.
- Preprocess the dataset by handling missing values and encoding categorical variables.
- Perform multiple iterations of feature importance analysis using random subsets of features.
- Save detailed results of each iteration to an output file.
- Generate a bar chart visualizing the average contribution of each feature.

Dependencies:
- pandas, numpy, matplotlib, sklearn

Functions:
- analyze_file: Performs feature importance analysis and saves iteration results.
- analyze_analize: Reads iteration results, calculates average contributions, and generates a bar chart.

Usage:
- Run the script directly to analyze a dataset and generate results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


def analyze_file(file_path: str, output_file: str, iteration: int):
    """
    Perform feature importance analysis on a dataset and save iteration results.

    Args:
        file_path (str): Path to the CSV file to be analyzed.
        output_file (str): Path to the file where iteration results will be saved.
        iteration (int): Number of iterations for feature importance analysis.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Preprocessing
    df = df.drop(columns=[col for col in ['ID', 'Unnamed: 0'] if col in df.columns])
    df.fillna(0, inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Features
    target_column = 'Price'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    results = []

    for i in range(iteration):  # Run iterations
        selected_columns = np.random.choice(X.columns, size=6, replace=False)
        X_subset = X[selected_columns]

        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate importance percentages for the selected columns
        column_importances = model.feature_importances_
        column_importance_percentages = (column_importances / column_importances.sum()) * 100

        results.append({
            'iteration': i + 1,
            'selected_columns': selected_columns.tolist(),
            'column_importance_percentages': column_importance_percentages.tolist(),
            'mse': mse,
            'r2': r2
        })

        print(f"Iteration {i + 1}:")
        print(f"Selected Columns: {selected_columns}")
        print(f"Column Importance Percentages: {column_importance_percentages}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R^2 Score: {r2:.2f}\n")

    # Save results to the output file
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"Iteration {result['iteration']}:\n")
            f.write(f"Selected Columns: {result['selected_columns']}\n")
            f.write(f"Column Importance Percentages: {result['column_importance_percentages']}\n")
            f.write(f"Mean Squared Error: {result['mse']:.2f}\n")
            f.write(f"R^2 Score: {result['r2']:.2f}\n\n")


def analyze_analize(file_path: str, output_file: str, output_file_png: str):
    """
    Read iteration results, calculate average contributions, and generate a bar chart.

    Args:
        file_path (str): Path to the file containing iteration results.
        output_file (str): Path to the file where average contributions will be saved.
        output_file_png (str): Path to the PNG file where the bar chart will be saved.
    """
    # Read the output file from analyze_file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    column_contributions = {}
    total_iterations = 0

    for line in lines:
        if line.startswith("Selected Columns:"):
            selected_columns = eval(line.split(": ")[1].strip())
        elif line.startswith("Column Importance Percentages:"):
            percentages = eval(line.split(": ")[1].strip())
            for col, perc in zip(selected_columns, percentages):
                column_contributions[col] = column_contributions.get(col, 0) + perc
            total_iterations += 1

    # Calculate average percentage contribution for each column
    average_contributions = {col: total / total_iterations for col, total in column_contributions.items()}

    # Ensure the order of columns matches the CSV file
    with open(file_path, 'r') as f:
        csv_columns = [line.split(": ")[1].strip() for line in lines if line.startswith("Selected Columns:")][0]
        csv_columns = eval(csv_columns)

    ordered_contributions = {col: average_contributions.get(col, 0) for col in csv_columns}

    # Generate a bar plot using matplotlib
    plt.figure(figsize=(10, 6))
    columns = list(ordered_contributions.keys())
    contributions = list(ordered_contributions.values())
    plt.bar(columns, contributions, color='skyblue')
    plt.xlabel('Columns')
    plt.ylabel('Average Contribution (%)')
    plt.title('Average Percentage Contribution of Each Column')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(output_file_png)
    plt.show()

    # Write the results to the output file
    with open(output_file, 'w') as f:
        f.write("Average Percentage Contribution of Each Column:\n")
        for col, avg in sorted(ordered_contributions.items(), key=lambda x: csv_columns.index(x[0])):
            f.write(f"{col}: {avg:.2f}%\n")


if __name__ == "__main__":
    """
    Main script execution.

    This section defines the input file, output files, and runs the analysis functions.
    """
    file_path = r'C:\Users\barte\Documents\LM\Cars\uae_used_cars_10k.csv'

    file_to_analyze = 'example_prediction.csv'
    analyze_file(file_path, file_to_analyze, 50)

    analyze_analize(file_to_analyze, 'finish_prediction.csv', 'finish_prediction.png')