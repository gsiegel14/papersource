#!/usr/bin/env python3
import os
import tarfile
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Compress VEXUS dataset directory into tarball')
    parser.add_argument('--source_dir', type=str, default='/Applications/VEXUS FINAL/VEXUS_Dataset_synthetic',
                        help='Path to the VEXUS dataset directory')
    parser.add_argument('--output_path', type=str, default='/Users/gabe/vexus_dataset.tar.gz',
                        help='Output path for the compressed tarball')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Compress the directory
    print(f"Compressing {args.source_dir} to {args.output_path}...")
    with tarfile.open(args.output_path, "w:gz") as tar:
        # Get the base directory name
        base_dir = os.path.basename(args.source_dir)
        # Change to the parent directory
        os.chdir(os.path.dirname(args.source_dir))
        # Add all files to the tarball
        tar.add(base_dir)
    
    size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"Successfully compressed to {args.output_path} ({size_mb:.2f} MB)")
    print(f"You can now run: python upload_dataset.py --file_path {args.output_path}")

if __name__ == "__main__":
    main() 