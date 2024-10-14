import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn.functional as F  # Import functional module for softmax

import torch

class ImageClassifier:
    def __init__(self, model_name='lenet', num_classes=10, input_image_size=(32, 32), transform=None, device='cpu'):
        """
        Initialize the classifier with a specific model name and hyperparameters.
        
        Parameters:
        - model_name: str, the name of the model (e.g., 'lenet', 'convnext_base', etc.).
        - num_classes: int, the number of output classes for the model.
        - input_image_size: tuple, input image dimensions (height, width).
        - transform: torchvision.transforms, transformations applied to the input images.
        - device: str, the device to use ('cpu' or 'cuda').
        """
        self.model_name=model_name
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.input_image_size = input_image_size
        self.transform = transform
        self.my_create_model()

    def my_create_model(self):

        # Dynamically select the model
        if self.model_name == 'lenet':
            self.model = LeNet(self.num_classes)
        else:
            self.model = getattr(models, self.model_name)(pretrained=True)  # Use models from torchvision
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)  # Modify final layer for custom num_classes
        
        self.model.to(self.device)

    def train(self, train_loader, epochs, learning_rate=0.001):
        """Train the model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            if epochs<10:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
            if (epoch%(epochs//10))==0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    def evaluate(self, test_loader):
        """Evaluate the model."""
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy

    def save(self, save_path):
        """Save the model state and hyperparameters to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'input_image_size': self.input_image_size,
            'transform': self.transform,
        }, save_path)
        print(f"Model and hyperparameters saved to {save_path}")

    def load(self, save_path):
        """Load the model state and hyperparameters from a file."""
        checkpoint = torch.load(save_path)
        
        self.num_classes = checkpoint['num_classes']
        self.input_image_size = checkpoint['input_image_size']
        self.transform = checkpoint.get('transform', None)  # Optional: Load transform if it exists
        self.my_create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model and hyperparameters loaded from {save_path}")
    
    def predict(self, pil_image):
        self.model.eval()
        image = self.transform(pil_image).unsqueeze(0).to(self.device)  # Transform and add batch dimension
        with torch.no_grad():
            outputs = self.model(image)  # Get raw model outputs (logits)
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            confidence, predicted = torch.max(probabilities, 1)  # Get class with max probability

        return predicted.item(), confidence.item()  # Return class_id and confidence level


# Define LeNet architecture
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import Counter

# Custom Dataset class to load images and labels from JSON annotations
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, augmentations=None):
        """
        Initialize the dataset with the annotations file and image directory.

        Parameters:
        - annotations_file: str, path to the JSON file containing image annotations.
        - img_dir: str, directory where images are stored.
        - transform: torchvision.transforms, transformations to apply to the images.
        - augmentations: torchvision.transforms, augmentations for data augmentation.
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        
        self.img_dir = img_dir
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get an image and its corresponding label by index.

        Parameters:
        - idx: int, index of the image-label pair.

        Returns:
        - tuple: (transformed image, label).
        """
        img_path = os.path.join(self.img_dir, self.annotations[idx]['image_file'])
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations[idx]['class_id'])

        # Apply augmentations (if any)
        if self.augmentations:
            image = self.augmentations(image)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, label


def set_seed(seed):
    """
    Set seed for reproducibility across all relevant libraries.
    
    Parameters:
    - seed: int, the seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(annotations_file):
    """
    Compute class weights based on class frequency for balancing the dataset.

    Parameters:
    - annotations_file: str, path to the JSON file containing image annotations.

    Returns:
    - list: class weights for each class.
    """
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)['annotations']
    
    labels = [int(ann['class_id']) for ann in annotations]
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Create a weight list corresponding to each sample
    sample_weights = [class_weights[label] for label in labels]
    return sample_weights


def load_data(annotations_file, img_dir, transform, augmentations=None, train_ratio=0.8, random_seed=42, batch_size=32):
    """
    Load the dataset, split it into train and test sets, and return the data loaders.

    Parameters:
    - annotations_file: str, path to the JSON file containing image annotations.
    - img_dir: str, directory where images are stored.
    - train_ratio: float, ratio of data to be used for training (between 0 and 1).
    - random_seed: int, seed for random splitting.
    - batch_size: int, number of samples per batch.

    Returns:
    - tuple: (train_loader, test_loader), DataLoader objects for train and test sets.
    """
    set_seed(random_seed)

    # Create the full dataset
    dataset = CustomImageDataset(annotations_file, img_dir, transform, augmentations)

    # Calculate the number of samples for train and test sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Randomly split the dataset
    generator = torch.Generator().manual_seed(random_seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)

    # Compute sample weights for the training set for resampling
    sample_weights = compute_class_weights(annotations_file)
    train_weights = [sample_weights[i] for i in train_set.indices]  # Apply weights to the train set
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader