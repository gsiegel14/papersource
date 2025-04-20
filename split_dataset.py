import os
import shutil
import random
from pathlib import Path
import argparse

def create_dataset_split(source_dir, dest_dir, train_ratio=0.7, seed=42):
    """
    Split a dataset into training and validation sets.
    
    Parameters:
    -----------
    source_dir : str
        Path to the source directory containing the dataset
    dest_dir : str
        Path to the destination directory for the split dataset
    train_ratio : float
        Ratio of data to use for training (default: 0.7)
    seed : int
        Random seed for reproducibility (default: 42)
    """
    random.seed(seed)
    
    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all category folders
    category_dirs = [d for d in os.listdir(source_dir) 
                   if os.path.isdir(os.path.join(source_dir, d)) 
                   and not d.startswith('.')]
    
    # Process each category
    for category in category_dirs:
        print(f"Processing category: {category}")
        
        # Create category directories in train and val
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        
        # Get all image files in the category
        category_path = os.path.join(source_dir, category)
        image_files = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                     and not f.startswith('.')]
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split
        train_size = int(len(image_files) * train_ratio)
        train_files = image_files[:train_size]
        val_files = image_files[train_size:]
        
        # Copy files to respective directories
        for file in train_files:
            src = os.path.join(category_path, file)
            dst = os.path.join(train_dir, category, file)
            shutil.copy2(src, dst)
        
        for file in val_files:
            src = os.path.join(category_path, file)
            dst = os.path.join(val_dir, category, file)
            shutil.copy2(src, dst)
        
        print(f"  - {len(train_files)} training images")
        print(f"  - {len(val_files)} validation images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into training and validation sets')
    parser.add_argument('--source', required=True, help='Source directory containing the dataset')
    parser.add_argument('--dest', required=True, help='Destination directory for the split dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Splitting dataset from {args.source} to {args.dest} with {args.train_ratio*100}% for training")
    create_dataset_split(args.source, args.dest, args.train_ratio, args.seed)
    print("Done!") 