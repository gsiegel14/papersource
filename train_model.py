#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import argparse
from pathlib import Path

# Define arguments
parser = argparse.ArgumentParser(description='Train a model on the VEXUS ultrasound dataset')
parser.add_argument('--data_dir', type=str, default='/Users/gabe/papersource/split_dataset_output',
                   help='Path to the split dataset directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--save_path', type=str, default='/Users/gabe/papersource/models',
                   help='Path to save the trained model')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
args = parser.parse_args()

# Create save directory if it doesn't exist
os.makedirs(args.save_path, exist_ok=True)

# Set device
device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
print(f"Using device: {device}")

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"Dataset classes: {class_names}")
print(f"Training set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")

# Load a pre-trained model and modify it for our task
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training function
def train_model(model, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    best_model_path = os.path.join(args.save_path, 'best_model.pth')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model with accuracy {best_acc:.4f}')
        
        print()
    
    # Save the final model
    final_model_path = os.path.join(args.save_path, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Training complete. Best val accuracy: {best_acc:.4f}')
    print(f'Best model saved to {best_model_path}')
    print(f'Final model saved to {final_model_path}')
    
    return model

# Train the model
if __name__ == '__main__':
    print("Starting training...")
    model = train_model(model, criterion, optimizer, num_epochs=args.epochs)
    print("Training completed!") 