"""
Created on Thu Mar 13 2025

@author: Juan Avendaño

Apply sequential Bayesian approach. Takes csv files with the results from the different models and combine them using
a bayesian approach in which the results of a model are used as the prior for the next one and then posteriors are
calculated.
"""


import numpy as np
import pandas as pd
from Metrics import calculate_metrics, plot_confusion_matrix

# ======================================================================================================================
# 0. Inputs & Configuration
# ======================================================================================================================
mode = "evaluation"  # Change to "prediction" for unlabeled case studies

probs_VGG16_path="C:\\Users\\jcac\\OneDrive - KTH\\Python\\CNN\\Journal-2\\Results\\00_Probabilities\\VGG16\\"
probs_Alex_path="C:\\Users\\jcac\\OneDrive - KTH\\Python\\CNN\\Journal-2\\Results\\00_Probabilities\\AlexNet\\"
probs_ResNet_path="C:\\Users\\jcac\\OneDrive - KTH\\Python\\CNN\\Journal-2\\Results\\00_Probabilities\\ResNet50\\"
results_path = r"C:\Users\jcac\OneDrive - KTH\Python\CNN\Journal-2\Results\02_Method_results\03_Sequential_Bayesian\\"

# ======================================================================================================================
# 1. Load probs csvs
# ======================================================================================================================
def parse_custom_csv(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=1)  # No headers assumed
    split_data = data[0].str.split(";", expand=True)

    # Assign columns
    split_data.columns = ['Image', 'Crack', 'Efflorescence', 'Spalling', 'Undamaged', 'True Label']

    for col in ['Crack', 'Efflorescence', 'Spalling', 'Undamaged', 'True Label']:
        split_data[col] = pd.to_numeric(split_data[col], errors='coerce')

    return split_data

csvVGG16 = probs_VGG16_path+"VGG16_test_probs.csv"
csvAlexNet = probs_Alex_path+"AlexNet_test_probs.csv"
csvResNet = probs_ResNet_path+"ResNet_test_probs.csv"

df1 = parse_custom_csv(csvVGG16)
df2 = parse_custom_csv(csvAlexNet)
df3 = parse_custom_csv(csvResNet)


# ======================================================================================================================
# 2. class probabilities
# ======================================================================================================================

# Assert all image names match
assert all(df1['Image'] == df2['Image']) and all(df1['Image'] == df3['Image']), "Image order mismatch!"

# Extract data
image_names = df1['Image'].tolist()
true_labels = df1['True Label'].tolist()
class_names = ['Crack', 'Efflorescence', 'Spalling', 'Undamaged']

# Extract class probabilities
M1_probs = df1[class_names].values
M2_probs = df2[class_names].values
M3_probs = df3[class_names].values

# ======================================================================================================================
# 3. Sequential Bayesian
# ======================================================================================================================
def bayesian_ensemble(probs_M1, probs_M2, probs_M3):

    M1_probs = np.array(probs_M1)
    M2_probs = np.array(probs_M2)
    M3_probs = np.array(probs_M3)

    prior = M1_probs

    likelihood_2 = M2_probs

    marginal_likelihood_mod2 = np.sum(likelihood_2*prior, axis=0)

    posterior_2= likelihood_2*prior

    posterior_mod2 = posterior_2/marginal_likelihood_mod2

    prior_2 = posterior_mod2
    likelihood_3 = M3_probs

    marginal_likelihood_mod3 = np.sum(likelihood_3 * prior_2, axis=0)

    posterior_3 = likelihood_3 * prior_2
    final_posterior = posterior_3/marginal_likelihood_mod3

    return final_posterior

# ======================================================================================================================
# 4. Run Inference & Evaluation
# ======================================================================================================================
# Main Script
if __name__ == "__main__":
    if mode == "evaluation":
        final_probs = [bayesian_ensemble(m1, m2, m3) for m1, m2, m3 in zip(M1_probs, M2_probs, M3_probs)]

        final_df = pd.DataFrame(final_probs, columns=class_names)
        final_df.insert(0, "Image", image_names)  # Add image names
        final_df["True Label"] = true_labels  # Add true labels

        # Get predicted class indices from final_probs
        predictions = np.argmax(final_df[class_names].values, axis=1)
        num_classes = len(class_names)

        metrics = calculate_metrics( predictions, true_labels, num_classes)
        plot_confusion_matrix(true_labels, predictions, class_names, "Bayesian")

        # Save results
        final_df.to_csv(results_path+"bayesian_fusion_results.csv", index=False)
        print("Final Bayesian predictions saved to 'bayesian_fusion_results.csv'")

    else:  # Prediction mode
        final_probs = [bayesian_ensemble(m1, m2, m3) for m1, m2, m3 in zip(M1_probs, M2_probs, M3_probs)]

        final_df = pd.DataFrame(final_probs, columns=class_names)
        final_df.insert(0, "Image", image_names)  # Add image names
        final_df["True Label"] = true_labels  # Add true labels

        # Save results
        final_df.to_csv(results_path + "bayesian_fusion_results.csv", index=False)
        print("Final Bayesian predictions saved to 'bayesian_fusion_results.csv'")