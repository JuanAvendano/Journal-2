"""
Created on Thu Apr 08 2025

@author: Juan Avendaño

Compare the metrics obtained using the different approaches of ensemble learning
"""
import os
import pandas as pd
from Metrics import calculate_metrics,plot_confusion_matrix
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import csv

def parse_custom_csv(file_path):
    with open(file_path, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        data = pd.read_csv(f, header=None, skiprows=1, sep=dialect.delimiter)
    # split_data = data[0].str.split(";", expand=True)

    # Assign columns
    data.columns = ['Image', 'Crack', 'Efflorescence', 'Spalling', 'Undamaged', 'True Label']

    for col in ['Crack', 'Efflorescence', 'Spalling', 'Undamaged', 'True Label']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    return data

def plot_metric_comparison(summary_metrics_df):
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"]

    summary_metrics_df.set_index("Method")[metrics_to_plot].plot(kind="bar", figsize=(10, 6))
    plt.title("Performance Comparison of Fusion Methods")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

# ======================================================================================================================
# 0. Inputs & Configuration
# ======================================================================================================================

results_path = r"C:\Users\jcac\OneDrive - KTH\Python\CNN\Journal-2\Results\02_Method_results\\"
metrics_path = r"C:\Users\jcac\OneDrive - KTH\Python\CNN\Journal-2\Results\03_Comparison_Metrics\\"

# ======================================================================================================================
# 1. Load csvs
# ======================================================================================================================

# load the different csv files for the different methods
method_dataframes = {}
method_names=[]
for sub_folder in os.listdir(results_path):
    sub_folder_path = os.path.join(results_path, sub_folder)
    if sub_folder.startswith('00_'):
        continue
    if os.path.isdir(sub_folder_path):
        # Remove leading numbers and underscores (e.g., 01_, 02_, etc.)
        cleaned_name = re.sub(r'^\d+_', '', sub_folder)
        method_names.append(cleaned_name)
        var_name = f"{cleaned_name}_csv"

    for file in os.listdir(sub_folder_path):
        csv_path = os.path.join(sub_folder_path, file)
        df =  parse_custom_csv(csv_path)
        method_dataframes[cleaned_name] = df
        break

# ======================================================================================================================
# 3. Metrics
# ======================================================================================================================
class_names = ['Crack', 'Efflorescence', 'Spalling', 'Undamaged']
results_summary = []
per_class_metrics = []

for method, df in method_dataframes.items():
    num_classes = len(class_names)
    image_names = df['Image'].tolist()
    true_labels = df['True Label'].tolist()
    predictions = np.argmax(df[class_names].values, axis=1)

    metrics = calculate_metrics(predictions, true_labels, num_classes)
    results_summary.append({"Method": method,
                            "Accuracy": metrics[0],
                            "Precision": metrics[1],
                            "Recall": metrics[2],
                            "F1 Score": metrics[3],
                            "F2 Score": metrics[5],
                            "Specificity (avg)": np.mean(metrics[4])})

    # 3.1 Per class metrics
    # =============================================================================================================
    precisions = precision_score(true_labels, predictions, average=None)
    recalls = recall_score(true_labels, predictions, average=None)
    f1s = f1_score(true_labels, predictions, average=None)
    for class_index, class_name in enumerate(class_names):
        per_class_metrics.append({
            "Method": method,
            "Class": class_name,
            "Precision": precisions[class_index],
            "Recall": recalls[class_index],
            "F1 Score": f1s[class_index],
            "Specificity": metrics[4][class_index]})

# ======================================================================================================================
# 4. Results saving
# ======================================================================================================================

summary_metrics_df = pd.DataFrame(results_summary)
summary_metrics_df.to_csv(metrics_path + "metrics_summary.csv", index=False)
print("Metrics summary saved.")

per_class_metrics_df = pd.DataFrame(per_class_metrics)
per_class_metrics_df.to_csv(os.path.join(metrics_path, "per_class_metrics.csv"), index=False)
print("Per-class specificity saved to per_class_metrics.csv")

# ======================================================================================================================
# 5. Comparison plot
# ======================================================================================================================

plot_metric_comparison(summary_metrics_df)