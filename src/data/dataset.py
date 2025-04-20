import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(".")
from configs.training_config import SDXLTrainingConfig

class UltrasoundDataset(Dataset):
    """Dataset for ultrasound pulse wave Doppler images"""
    
    def __init__(
        self,
        image_paths: List[str],
        class_labels: List[str],
        tokenizer=None,
        image_size: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True
    ):
        self.image_paths = image_paths
        self.class_labels = class_labels
        self.tokenizer = tokenizer
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        class_label = self.class_labels[idx]
        
        # Create prompts based on class - these will be used for text conditioning
        prompt = f"A pulse wave Doppler ultrasound image showing {class_label.replace('_', ' ')}"
        
        # Tokenize prompt if tokenizer is provided
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            return {
                "image": image,
                "input_ids": text_inputs.input_ids[0],
                "class_label": class_label,
            }
        
        return {
            "image": image,
            "prompt": prompt,
            "class_label": class_label,
        }


def prepare_dataset(config: SDXLTrainingConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare training, validation, and test datasets from the ultrasound images
    
    Args:
        config: Training configuration
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    image_paths = []
    class_labels = []
    
    # Collect all image paths and corresponding class labels
    for class_name in config.classes:
        class_dir = os.path.join(config.dataset_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory {class_dir} not found")
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                image_path = os.path.join(class_dir, filename)
                image_paths.append(image_path)
                class_labels.append(class_name)
    
    # Ensure data was found
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {config.dataset_path}")
    
    print(f"Found {len(image_paths)} images across {len(config.classes)} classes")
    
    # Create a combined list and shuffle with same seed
    combined = list(zip(image_paths, class_labels))
    random.seed(42)
    random.shuffle(combined)
    image_paths, class_labels = zip(*combined)
    
    # Split the data
    total_size = len(image_paths)
    train_size = int(total_size * config.train_split_ratio)
    val_size = int(total_size * config.validation_split_ratio)
    
    train_image_paths = image_paths[:train_size]
    train_class_labels = class_labels[:train_size]
    
    val_image_paths = image_paths[train_size:train_size + val_size]
    val_class_labels = class_labels[train_size:train_size + val_size]
    
    test_image_paths = image_paths[train_size + val_size:]
    test_class_labels = class_labels[train_size + val_size:]
    
    print(f"Train set: {len(train_image_paths)} images")
    print(f"Validation set: {len(val_image_paths)} images")
    print(f"Test set: {len(test_image_paths)} images")
    
    # Create datasets
    train_dataset = UltrasoundDataset(train_image_paths, train_class_labels, image_size=config.image_size)
    val_dataset = UltrasoundDataset(val_image_paths, val_class_labels, image_size=config.image_size, random_flip=False)
    test_dataset = UltrasoundDataset(test_image_paths, test_class_labels, image_size=config.image_size, random_flip=False)
    
    return train_dataset, val_dataset, test_dataset


def get_data_loaders(config: SDXLTrainingConfig):
    """Get data loaders for training, validation, and test sets"""
    train_dataset, val_dataset, test_dataset = prepare_dataset(config)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.train_batch_size * 2,
        shuffle=False,
        num_workers=4,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.train_batch_size * 2,
        shuffle=False,
        num_workers=4,
    )
    
    return train_dataloader, val_dataloader, test_dataloader 