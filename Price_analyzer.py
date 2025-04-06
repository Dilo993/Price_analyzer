"""
Price Analyzer Script

This script provides a GUI-based tool for analyzing the contribution of various features
(columns) in a dataset to a target variable (e.g., price). It uses a Random Forest Regressor
to calculate feature importance and visualizes the results in a bar chart.

Key Features:
- Load a CSV file through a file dialog.
- Preprocess the dataset by handling missing values and encoding categorical variables.
- Perform multiple iterations of feature importance analysis using random subsets of features.
- Display a progress bar during the analysis.
- Visualize the average contribution of each feature in a bar chart.

Dependencies:
- pandas, numpy, matplotlib, tkinter, sklearn, threading
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import threading

def analyze_combined(file_path: str, iteration: int):
    """
    Perform feature importance analysis on a dataset.

    Args:
        file_path (str): Path to the CSV file to be analyzed.
        iteration (int): Number of iterations for feature importance analysis.
    """
    def run_analysis():
        """
        Run the analysis in a separate thread to avoid blocking the GUI.

        This function loads the dataset, preprocesses it, performs feature importance
        analysis using a Random Forest Regressor, and visualizes the results in a bar chart.
        """
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestRegressor
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        """
        Load the CSV file and set the window title.
        """
        root.title("Please pick")
        df = pd.read_csv(file_path)

        """
        Preprocess the dataset by dropping unnecessary columns and handling missing values.
        """
        df = df.drop(columns=[col for col in ['ID', 'Unnamed: 0'] if col in df.columns])
        df.fillna(0, inplace=True)

        """
        Encode categorical variables using LabelEncoder.
        """
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

        """
        Identify target columns and ensure at least one exists in the dataset.
        """
        target_columns = ['Price', 'price', 'Prices', 'price']
        existing_target_columns = [col for col in target_columns if col in df.columns]

        if not existing_target_columns:
            raise ValueError("None of the target columns exist in the DataFrame.")

        """
        Separate features (X) and target (y) for analysis.
        """
        X = df.drop(columns=existing_target_columns)
        y = df[existing_target_columns]

        column_contributions = {}
        total_iterations = 0

        """
        Create and display a progress bar for tracking analysis progress.
        """
        progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        progress.place(relx=0.5, rely=0.5, anchor='center')
        progress['maximum'] = iteration

        """
        Preserve the original column order for consistent visualization.
        """
        original_column_order = X.columns.tolist()

        """
        Perform multiple iterations of feature importance analysis.
        """
        root.title("Calculating...")
        for i in range(iteration):
            selected_columns = np.random.choice(X.columns, size=min(6, len(X.columns)), replace=False)
            X_subset = X[selected_columns]

            """
            Split the data into training and testing sets.
            """
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

            """
            Train a Random Forest Regressor and calculate feature importance.
            """
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            column_importances = model.feature_importances_
            column_importance_percentages = (column_importances / column_importances.sum()) * 100

            """
            Accumulate contributions for each column across iterations.
            """
            for col, perc in zip(selected_columns, column_importance_percentages):
                column_contributions[col] = column_contributions.get(col, 0) + perc
            total_iterations += 1

            """
            Update the progress bar to reflect the current iteration.
            """
            progress['value'] = i + 1
            root.update_idletasks()

        """
        Remove the progress bar and restore the file selection button.
        """
        progress.destroy()
        open_file_button.place(relx=0.5, rely=0.5, anchor='center')

        """
        Calculate average contributions and sort them for visualization.
        """
        average_contributions = {col: total / total_iterations for col, total in column_contributions.items()}
        sorted_contributions = {col: average_contributions[col] for col in original_column_order if col in average_contributions}

        """
        Create a bar chart to visualize the average contributions.
        """
        fig = plt.figure(figsize=(10, 6))
        columns = list(sorted_contributions.keys())
        contributions = list(sorted_contributions.values())
        plt.bar(columns, contributions, color='skyblue')
        plt.xlabel('Columns')
        plt.ylabel('Average Contribution (%)')
        plt.title('Average Percentage Contribution of Each Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        """
        Embed the bar chart in the Tkinter window.
        """
        root.title("Analysis Results")
        window_width, window_height = fig.get_size_inches() * fig.dpi
        root.geometry(f"{int(window_width)}x{int(window_height)}")

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        """
        Add a close button to terminate the application.
        """
        tk.Button(root, text="Close", command=root.destroy).pack()

    """
    Start the analysis in a separate thread to keep the GUI responsive.
    """
    analysis_thread = threading.Thread(target=run_analysis, daemon=True)
    analysis_thread.start()

if __name__ == "__main__":
    """
    Initialize the Tkinter root window and configure its properties.
    """
    root = tk.Tk()
    root.geometry("800x600")
    root.resizable(False, False)

    """
    Center the window on the screen.
    """
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 800
    window_height = 600
    position_top = int((screen_height - window_height) / 2)
    position_right = int((screen_width - window_width) / 2)
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    """
    Ensure the main loop stops when the window is closed.
    """
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    def open_file():
        """
        Open a file dialog to select a CSV file and start the analysis.
        """
        file_path = filedialog.askopenfilename(title="Select a CSV File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            open_file_button.place_forget()
            analyze_combined(file_path, 50)
        else:
            print("No file selected.")

    """
    Add a button to open the file dialog for selecting a CSV file.
    """
    open_file_button = tk.Button(root, text="Open File", command=open_file)
    open_file_button.place(relx=0.5, rely=0.5, anchor='center')

    """
    Bind the Enter key to close the window.
    """
    root.bind('<Return>', lambda event: root.destroy())

    """
    Start the Tkinter main loop to display the GUI.
    """
    root.mainloop()