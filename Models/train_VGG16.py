"""
Created on Fri Feb 21 2025

@author: Juan

CNN for training VGG16
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import os


# Define training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs, save_path):
    best_acc = 0.0
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in dataloaders['train']:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += labels.size(0)
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_acc = correct_preds.double() / total_preds
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at {save_path}")


# Evaluation function
def evaluate_model(model, dataloaders, criterion):
    model.eval()
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

    return accuracy, precision, recall, f1


# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VGG16 for multi-damage detection")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of damage classes")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--save_model', type=str, default="vgg16_best.pth", help="Path to save the best model")
    args = parser.parse_args()

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    }

    # Load VGG16 model
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.num_classes)

    if torch.cuda.is_available():
        model.cuda()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(model, dataloaders, criterion, optimizer, args.epochs, args.save_model)

    # Load best model
    model.load_state_dict(torch.load(args.save_model))

    # Evaluate model
    evaluate_model(model, dataloaders, criterion)
