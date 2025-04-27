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

"""
Add a global flag to signal the thread to stop
"""
stop_analysis = False

def analyze_combined(file_path: str, iteration: int, target_column: str):
    global stop_analysis

    """
    Perform feature importance analysis on a dataset.

    Args:
        file_path (str): Path to the CSV file to be analyzed.
        iteration (int): Number of iterations for feature importance analysis.
        target_column (str): The column to be used as the target variable.
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
        Load the CSV file and update the window title.
        """
        root.title("Please pick")
        df = pd.read_csv(file_path)
        root.title("Calculating...")

        """
        Drop unnecessary columns and handle missing values.
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
        Ensure the selected target column exists in the dataset.
        """
        if target_column not in df.columns:
            raise ValueError(f"The selected target column '{target_column}' does not exist in the DataFrame.")

        """
        Separate features (X) and target (y) for analysis.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column].values.ravel()

        column_contributions = {}
        total_iterations = 0

        """
        Create and display a progress bar for the analysis.
        """
        progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        progress.place(relx=0.5, rely=0.5, anchor='center')
        progress['maximum'] = iteration

        """
        Preserve the original column order.
        """
        original_column_order = X.columns.tolist()

        """
        Perform multiple iterations of feature importance analysis.
        """
        def update_progress(value):
            """
            Update the progress bar in the main thread.
            """
            progress['value'] = value
            root.update_idletasks()

        def show_results():
            """
            Display the results in the main thread.
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
            Embed the bar chart in the Tkinter window and add a close button.
            """
            root.title("Analysis Results")
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            """
            Add a close button to terminate the application
            """
            tk.Button(root, text="Close", command=root.destroy).pack()

        for i in range(iteration):
            if stop_analysis:
                break

            """
            Shuffle and split columns for fair selection
            """
            shuffled_columns = np.random.permutation(X.columns)
            selected_columns = shuffled_columns[:max(1, len(X.columns) // 2)]
            X_subset = X[selected_columns]

            """
            Split the data into training and testing sets
            """
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

            """
            Train a Random Forest Regressor and calculate feature importance
            """
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            column_importances = model.feature_importances_
            column_importance_percentages = (column_importances / column_importances.sum()) * 100

            """
            Accumulate contributions for each column
            """
            for col, perc in zip(selected_columns, column_importance_percentages):
                column_contributions[col] = column_contributions.get(col, 0) + perc
            total_iterations += 1

            """
            Schedule progress bar update in the main thread
            """
            root.after(0, update_progress, i + 1)

        """
        Schedule results display in the main thread
        """
        root.after(0, show_results)

    """
    Start the analysis in a separate thread
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
    Center the window on the screen
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
    Ensure the main loop stops when the window is closed
    """
    def on_close():
        global stop_analysis
        stop_analysis = True 
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    """
    Define the function to open a file dialog and start the analysis.
    """
    def open_file():
        """
        Open a file dialog to select a CSV file and start the analysis.
        """
        file_path = filedialog.askopenfilename(title="Select a CSV File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            open_file_button.place_forget()
            """
            Load the CSV file to determine available columns
            """
            import pandas as pd
            df = pd.read_csv(file_path)
            """
            Create a dropdown menu to select the target column
            """
            target_var_label = tk.Label(root, text="Select Target Column:")
            target_var_label.place(relx=0.5, rely=0.4, anchor='center')

            target_var = tk.StringVar(root)
            target_var.set(df.columns[0])
            target_dropdown = tk.OptionMenu(root, target_var, *df.columns)
            target_dropdown.place(relx=0.5, rely=0.45, anchor='center')

            def start_analysis():
                """
                Start the analysis with the selected target column
                """
                target_var_label.destroy()
                target_dropdown.destroy()
                start_button.destroy()
                analyze_combined(file_path, 50, target_var.get())

            """
            Add a button to confirm the target column selection and start the analysis
            """
            start_button = tk.Button(root, text="Start Analysis", command=start_analysis)
            start_button.place(relx=0.5, rely=0.5, anchor='center')
        else:
            print("No file selected.")

    """
    Add a button to open the file dialog and start the Tkinter main loop.
    """
    open_file_button = tk.Button(root, text="Open File", command=open_file)
    open_file_button.place(relx=0.5, rely=0.5, anchor='center')

    """
    Start the Tkinter main loop
    """
    root.mainloop()