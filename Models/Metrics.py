"""
Created on Fri Mar 07 2025

@author: Juan Avendaño

Ensemble learning fusion by Hard Voting
"""
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(predictions, true_labels, num_classes):
    cm = confusion_matrix(true_labels, predictions, labels=list(range(num_classes)))

    # Accuracy
    accuracy = accuracy_score(true_labels, predictions)

    # Precision, Recall, F1 Score (per class)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    # Specificity (TN / (TN + FP)) per class
    specificity = np.diag(cm) / np.sum(cm, axis=1)

    # F2 Score (weighted)
    f2 = 5 * (precision * recall) / (4 * precision + recall)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity}")
    print(f"F2 Score: {f2:.4f}")

    return  accuracy, precision, recall, f1, specificity, f2


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Example of calling the function
# cm, accuracy, precision, recall, f1, specificity, f2 = calculate_metrics(predictions, true_labels)



