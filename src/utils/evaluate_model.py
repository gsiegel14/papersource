import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(".")
from configs.training_config import SDXLTrainingConfig
from src.data.dataset import get_data_loaders, UltrasoundDataset

class ImageFolderDataset(Dataset):
    """Dataset for loading images from folder"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.class_labels = []
        
        # Look for class subdirectories
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        if class_dirs:
            # If we have class subdirectories, use them
            for class_dir in class_dirs:
                class_path = os.path.join(root_dir, class_dir)
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                        self.image_paths.append(os.path.join(class_path, filename))
                        self.class_labels.append(class_dir)
        else:
            # If no class subdirectories, just load all images
            for filename in os.listdir(root_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.class_labels.append("unknown")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.class_labels[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated ultrasound images")
    parser.add_argument(
        "--real_data_dir", 
        type=str, 
        default="/Applications/VEXUS FINAL/VEXUS_Dataset_synthetic",
        help="Directory containing real ultrasound images"
    )
    parser.add_argument(
        "--generated_data_dir", 
        type=str, 
        default="./generated_images",
        help="Directory containing generated ultrasound images"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=299,
        help="Image size for evaluation (299 for FID with Inception model)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--eval_type", 
        type=str, 
        default="fid",
        choices=["fid", "visual"],
        help="Type of evaluation to perform"
    )
    return parser.parse_args()

def calculate_fid(real_dataloader, generated_dataloader, device):
    """Calculate Fréchet Inception Distance between real and generated images"""
    
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Add real images
    for images, _ in tqdm(real_dataloader, desc="Processing real images"):
        images = images.to(device)
        fid.update(images, real=True)
    
    # Add generated images
    for images, _ in tqdm(generated_dataloader, desc="Processing generated images"):
        images = images.to(device)
        fid.update(images, real=False)
    
    # Calculate FID
    fid_score = fid.compute()
    
    return fid_score.item()

def visualize_samples(real_dataloader, generated_dataloader, num_samples=5):
    """Visualize and compare real and generated samples"""
    
    # Get some random real samples
    real_iter = iter(real_dataloader)
    real_images, real_labels = next(real_iter)
    
    # Get some random generated samples
    gen_iter = iter(generated_dataloader)
    gen_images, gen_labels = next(gen_iter)
    
    # Plot real vs generated for each class
    plt.figure(figsize=(15, 10))
    
    # Get unique classes from the current batch
    unique_labels = set(real_labels[:num_samples])
    
    for i, label in enumerate(unique_labels):
        # Find indices for this class in real images
        real_idx = [j for j, l in enumerate(real_labels) if l == label]
        if not real_idx:
            continue
        real_idx = real_idx[0]
        
        # Find indices for this class in generated images
        gen_idx = [j for j, l in enumerate(gen_labels) if l == label]
        if not gen_idx:
            continue
        gen_idx = gen_idx[0]
        
        # Plot real image
        plt.subplot(len(unique_labels), 2, i*2+1)
        plt.imshow(real_images[real_idx].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # Denormalize
        plt.title(f"Real: {label}")
        plt.axis("off")
        
        # Plot generated image
        plt.subplot(len(unique_labels), 2, i*2+2)
        plt.imshow(gen_images[gen_idx].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # Denormalize
        plt.title(f"Generated: {label}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("sample_comparison.png")
    print("Sample comparison saved to sample_comparison.png")

def main():
    args = parse_args()
    config = SDXLTrainingConfig()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define transforms for evaluation
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create datasets
    real_dataset = ImageFolderDataset(args.real_data_dir, transform=transform)
    generated_dataset = ImageFolderDataset(args.generated_data_dir, transform=transform)
    
    # Create dataloaders
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    generated_dataloader = DataLoader(
        generated_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    print(f"Real dataset size: {len(real_dataset)}")
    print(f"Generated dataset size: {len(generated_dataset)}")
    
    # Evaluate based on evaluation type
    if args.eval_type == "fid":
        fid_score = calculate_fid(real_dataloader, generated_dataloader, device)
        print(f"FID Score: {fid_score}")
        
        # Save FID score to file
        with open("fid_results.txt", "w") as f:
            f.write(f"FID Score: {fid_score}\n")
            f.write(f"Real dataset size: {len(real_dataset)}\n")
            f.write(f"Generated dataset size: {len(generated_dataset)}\n")
    
    elif args.eval_type == "visual":
        visualize_samples(real_dataloader, generated_dataloader)

if __name__ == "__main__":
    main() 