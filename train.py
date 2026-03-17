import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import gc

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a custom loader that handles corrupted images
def pil_loader(path):
    try:
        from PIL import Image
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"Warning: Could not load image at {path}")
        print(f"Error: {str(e)}")
        return None

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"Error loading image at index {index}")
            # Try the next image
            if index + 1 < len(self):
                return self.__getitem__(index + 1)
            else:
                # If we're at the end, try from the beginning
                return self.__getitem__(0)

# Load datasets with custom handler
train_dataset = CustomImageFolder(root='DATASET/TRAIN', transform=transform, loader=pil_loader)
test_dataset = CustomImageFolder(root='DATASET/TEST', transform=transform, loader=pil_loader)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load ResNet50 model
model = models.resnet50(pretrained=True)

# Enable gradient checkpointing to reduce memory usage
model.train()
for module in model.modules():
    if isinstance(module, torch.nn.modules.conv.Conv2d):
        module.checkpoint = True

# Modify the final layer for our number of classes (5 yoga poses)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Add garbage collection at the start of each epoch
        gc.collect()
        torch.cuda.empty_cache()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Clear some memory (after we're done using outputs)
            del outputs
            torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {accuracy:.2f}%')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_accuracy = 100. * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%\n')

# Train the model
train_model(num_epochs=10)

# Save the model
torch.save(model.state_dict(), 'yoga_pose_classifier.pth')
