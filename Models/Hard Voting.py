"""
Created on Thu Mar 06 2025

@author: Juan Avendaño

Ensemble learning fusion by Hard Voting
"""
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
from VGG16 import load_VGG16
from ResNet50 import load_ResNet50
from AlexNet import load_AlexNet
from torchvision.datasets import ImageFolder
from Metrics import calculate_metrics
from Metrics import plot_confusion_matrix

# ======================================================================================================================
# 0. Inputs & Configuration
# ======================================================================================================================

mode = "evaluation"  # Change to "prediction" for unlabeled case studies

# image_path = r"C:\Users\jcac\OneDrive - KTH\Datasets\DataEnsemble\03-test\c9.jpg"
image_folder = "C:\\Users\\jcac\\OneDrive - KTH\\Datasets\\DataEnsemble\\02-test"

# class_names = sorted(os.listdir(image_folder))  # Extract class names from folder structure
class_names = {0: "Crack", 1: "Efflorescence", 2: "Spalling", 3: "Undamaged"}
num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================================================================
# 1. Load trained models
# ======================================================================================================================

# Load models
vgg16 = load_VGG16(num_classes, device)
resnet50 = load_ResNet50(num_classes, device)
alexnet = load_AlexNet(num_classes, device)

# Load trained weights
vgg16.load_state_dict(torch.load(r"C:\Users\jcac\OneDrive - KTH\Python\CNN\Journal-2\Saved_models\best_VGG16v2.pth", map_location=device))
resnet50.load_state_dict(torch.load(r"C:\Users\jcac\OneDrive - KTH\Python\CNN\Journal-2\Saved_models\best_ResNet50v01.pth", map_location=device))
alexnet.load_state_dict(torch.load(r"C:\Users\jcac\OneDrive - KTH\Python\CNN\Journal-2\Saved_models\best_AlexNetv01.pth", map_location=device))


# Set models to evaluation mode
vgg16.eval()
resnet50.eval()
alexnet.eval()

# Dictionary of models
models = {"VGG16": vgg16,"ResNet50": resnet50,"AlexNet": alexnet}

# ======================================================================================================================
# 2. Image Preprocessing & Dataloader
# ======================================================================================================================

def get_transform(model_name):
    size = 227 if model_name == "AlexNet" else 224

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to preprocess the image
def preprocess_image(image_path, model_name):
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
    transform = get_transform(model_name)
    image = transform(image)  # Apply transformation
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    return image


def load_images(image_folder, mode):
    image_paths, labels = [], []

    for class_idx, class_name in enumerate(class_names.values()):
        class_path = os.path.join(image_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            if mode == "evaluation":
                labels.append(class_idx)

    return image_paths, labels if mode == "evaluation" else image_paths

# ======================================================================================================================
# 3. Hard Voting Function
# ======================================================================================================================
# Hard Voting function
def hard_voting_ensemble(image_path, models):
    votes = []

    for model_name,model in models.items():
        image = preprocess_image(image_path, model_name)
        with torch.no_grad():
            outputs = model(image)
            predicted_class = torch.argmax(F.softmax(outputs, dim=1), dim=1).item()
            votes.append(predicted_class)


    # Majority voting
    final_prediction = max(set(votes), key=votes.count)
    return final_prediction

# ======================================================================================================================
# 4. Run Inference & Evaluation
# ======================================================================================================================
# Main Script
if __name__ == "__main__":

    if mode == "evaluation":
        image_paths, true_labels = load_images(image_folder, mode)
        predictions = [hard_voting_ensemble(img, models) for img in image_paths]

        metrics = calculate_metrics( predictions, true_labels, num_classes)
        plot_confusion_matrix(true_labels, predictions, class_names.values(), "Hard Voting")

    else:  # Prediction mode
        image_paths = load_images(image_folder, mode)
        predictions = [class_names[hard_voting_ensemble(img, models)] for img in image_paths]

        for img_path, pred in zip(image_paths, predictions):
            print(f"Image: {os.path.basename(img_path)} -> Predicted Class: {pred}")



# # Run hard voting ensemble
# final_label = hard_voting_ensemble(image_path,models)
# final_prediction = class_names[final_label]
#
# cm, accuracy, precision, recall, f1, specificity, f2 = calculate_metrics()
#
# print(f'Ensemble Prediction (Hard Voting): {final_prediction}')