"""
Created on Fri Feb 21 2025

@author: Juan  Avendaño

CNN for training VGG16
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import time
import os
from Metrics import calculate_metrics
from Metrics import plot_confusion_matrix

# ======================================================================================================================
# 0. Inputs
# ======================================================================================================================

train_path = "C:\\Users\\jcac\\OneDrive - KTH\\Datasets\\DataEnsemble\\01-train"   # Paths to train dataset
test_path = "C:\\Users\\jcac\\OneDrive - KTH\\Datasets\\DataEnsemble\\02-test"     # Paths to test dataset
save_path="C:\\Users\\jcac\\OneDrive - KTH\\Python\\CNN\\Journal-2\\Saved_models\\best_AlexNetv01.pth"
save_probs_path="C:\\Users\\jcac\\OneDrive - KTH\\Python\\CNN\\Journal-2\\Saved_models\\Probabilities\\AlexNet\\"
batch_size = 32     # Define batch size
split_ratio = 0.8   # Split ratio for training and validation
lrn_rate = 0.001
epochs = 50

# ======================================================================================================================
# 1. Functions
# ======================================================================================================================

def load_AlexNet(num_classes, device):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False  # Freeze all convolutional layers

    return model

# Define training function
def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs=epochs, save_path=save_path):
    """
    Train the model and save the best one based on validation accuracy.

    Parameters:
    - model: The model to be trained.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - loss_func: Loss function (e.g., CrossEntropyLoss).
    - optimizer: Optimizer (e.g., Adam).
    - num_epochs: Number of epochs to train the model.
    - save_path: Path to save the best model.
    """
    best_accuracy = 0.0  # To track the best model's accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)  # Move model to the appropriate device (GPU/CPU)

    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing the epoch
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate the training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Validation loop
        val_accuracy = evaluate_accuracy(model, val_loader, device)
        val_loss = evaluate_loss(model, val_loader, loss_func, device)

        epoch_time = time.time() - start_time  # Calculate time taken for the epoch

        # Print the results in the desired format
        print(
            f"Epoch [{epoch+1}/{num_epochs}],- {epoch_time:.0f}s - loss: {train_loss:.4f} - acc: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f}")

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_accuracy:
            print(f"Validation accuracy improved. Saving model...")
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)  # Save the best model

    print("Training complete.")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

# Helper function for evaluating the model's loss (e.g.,loss on the validation set)
def evaluate_loss(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():  # No gradients are needed for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(val_loader)

# Helper function for evaluating the model (e.g., accuracy on the validation set)
def evaluate_accuracy(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients are needed for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def evaluate_final_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    labels_list = []
    all_probs = []  # Store softmax probabilities
    filenames_list = []

    with torch.no_grad():  # No gradients are needed for testing
        for inputs, labels, paths in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels_list += labels.tolist()
            filenames_list += [os.path.basename(p) for p in paths]  # Store just the image name
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities

            # Save probabilities and labels
            all_probs.append(probs.cpu().numpy())

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # inference
            predictions.extend(predicted.cpu().numpy())

    metrics = calculate_metrics(predictions, labels_list, num_classes)
    plot_confusion_matrix(labels_list, predictions, class_names, "AlexNet")

    # Calculates test loss and accuracy
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

    # Convert lists to numpy arrays
    all_probs = np.concatenate(all_probs, axis=0)

    # Save probabilities and labels to file
    df_probs = pd.DataFrame(all_probs, columns=[f'Class_{i}_prob' for i in range(all_probs.shape[1])])
    df_probs.insert(0, 'Image_Name', filenames_list)
    df_probs['True Label'] = labels_list

    # Save DataFrame to CSV
    csv_file_path = os.path.join(save_probs_path, "AlexNet_probs.csv")
    df_probs.to_csv(csv_file_path, index=False)
    print(f"Saved probabilities to {csv_file_path}")

    return predictions  # Return list of predictions

# ======================================================================================================================
# 2. Loading and Preprocessing of Data
# ======================================================================================================================

#  ===== Data Transformations =====

# Resize and normalize pixel values in the input images
transform = transforms.Compose([
    transforms.Resize((227, 227)), # Resize to match AlexNet input size
    transforms.ToTensor(), # Convert images to PyTorch tensors. PIL or OpenCV organize images as Height, Width, Channels but Pytorch organize images as Channels, Height, Witdh so we change it to the correct format for Pytorch
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Values from the ImageNet dataset

# ===== Load Dataset =====

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        # Original tuple: (image_tensor, label)
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        # Return (image_tensor, label, path)
        return original_tuple + (path,)

# Load datasets
train_dataset = ImageFolderWithPaths(root=train_path, transform=transform)
test_dataset = ImageFolderWithPaths(root=test_path, transform=transform)

# Print class names to verify
print(train_dataset.classes)

# ===== Data Loaders =====

# Define split ratio
train_size = int(split_ratio * len(train_dataset))  # split_ratio value for training
val_size = len(train_dataset) - train_size  # remaining for validation

# Split dataset
train_data, val_data = random_split(train_dataset, [train_size, val_size])

# Create data loaders in batches of images
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check the number of images
print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_dataset)}")

# ======================================================================================================================
# 3. Implementation
# ======================================================================================================================
# Main Script
if __name__ == "__main__":
    # Initialize and set up data, model, etc.
    class_names = os.listdir(test_path)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_AlexNet(num_classes, device)
    # Define the loss function
    loss_func = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lrn_rate)  # Learning rate set to 0.001


    # More setup here ...
    # train_model(model, train_loader, val_loader, loss_func, optimizer,epochs)

    # Load the best model
    model.load_state_dict(torch.load(save_path))  # Make sure to load the model after training

    # Evaluate the model on the test set
    evaluate_final_model(model, test_loader, loss_func, device)



