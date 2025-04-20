#!/usr/bin/env python3
import os
import argparse
import shutil
import random
from pathlib import Path
import json

def create_splits(source_dir, output_dir, train_ratio=0.7, seed=42):
    """
    Split dataset into training and validation sets
    
    Args:
        source_dir: Path to source directory containing the dataset
        output_dir: Path to output directory where splits will be created
        train_ratio: Ratio of data to use for training (0.0-1.0)
        seed: Random seed for reproducibility
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Check if source directory exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create train and validation directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Track statistics
    stats = {
        "total_files": 0,
        "train_files": 0,
        "val_files": 0,
        "categories": {}
    }
    
    # Find all category directories in the source
    category_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    for category_dir in category_dirs:
        category_name = category_dir.name
        print(f"Processing category: {category_name}")
        
        # Create corresponding category directories in train and val
        (train_dir / category_name).mkdir(exist_ok=True)
        (val_dir / category_name).mkdir(exist_ok=True)
        
        # Get all files in the category
        files = [f for f in category_dir.glob("**/*") if f.is_file()]
        
        if not files:
            print(f"  No files found in category: {category_name}")
            continue
        
        # Shuffle files
        random.shuffle(files)
        
        # Split into train and validation
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        # Copy files to train directory
        for file in train_files:
            # Preserve directory structure relative to category
            rel_path = file.relative_to(category_dir)
            target_path = train_dir / category_name / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target_path)
        
        # Copy files to validation directory
        for file in val_files:
            # Preserve directory structure relative to category
            rel_path = file.relative_to(category_dir)
            target_path = val_dir / category_name / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target_path)
        
        # Update statistics
        stats["total_files"] += len(files)
        stats["train_files"] += len(train_files)
        stats["val_files"] += len(val_files)
        stats["categories"][category_name] = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files)
        }
        
        print(f"  Split {len(files)} files: {len(train_files)} train, {len(val_files)} validation")
    
    # Save statistics to JSON file
    with open(output_path / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset split complete!")
    print(f"Total files: {stats['total_files']}")
    print(f"Training files: {stats['train_files']} ({stats['train_files']/stats['total_files']*100:.1f}%)")
    print(f"Validation files: {stats['val_files']} ({stats['val_files']/stats['total_files']*100:.1f}%)")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Split dataset into training and validation sets')
    parser.add_argument('--source_dir', type=str, 
                        default='/Applications/VEXUS FINAL/VEXUS_Dataset_synthetic',
                        help='Directory containing the source dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='/Users/gabe/vexus_splits',
                        help='Directory to save the training and validation splits')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Print settings
    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Random seed: {args.seed}")
    
    try:
        stats = create_splits(
            args.source_dir,
            args.output_dir,
            args.train_ratio,
            args.seed
        )
        print("\nSplit statistics saved to split_stats.json")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 