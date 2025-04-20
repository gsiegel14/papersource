#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import json
import time

def run_command(command):
    """Run a command and return its output"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Error: {stderr}")
        sys.exit(1)
    
    return stdout.strip()

def upload_dataset(name, description, file_path, storage_provider_id):
    """Upload dataset to Gradient"""
    
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Dataset file size: {file_size_mb:.2f} MB")
    
    # Check if Gradient CLI is installed
    try:
        gradient_version = run_command("gradient version")
        print(f"Using Gradient CLI: {gradient_version}")
    except:
        print("Error: Gradient CLI not found. Please install it using 'pip install gradient'")
        sys.exit(1)
    
    # Create dataset in Gradient
    print(f"Creating dataset '{name}' in Gradient...")
    create_cmd = f'gradient datasets create --name "{name}" --description "{description}" --storageProviderId {storage_provider_id}'
    
    dataset_info = run_command(create_cmd)
    try:
        dataset_data = json.loads(dataset_info)
        dataset_id = dataset_data.get('id')
        if not dataset_id:
            print(f"Error: Could not parse dataset ID from response: {dataset_info}")
            sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON response: {dataset_info}")
        sys.exit(1)
        
    print(f"Dataset created with ID: {dataset_id}")
    
    # Upload dataset to storage provider
    print(f"Uploading dataset file to storage provider...")
    print(f"This may take a while depending on your internet connection and file size.")
    
    upload_cmd = f'gradient datasets files upload --id {dataset_id} --path {file_path}'
    upload_result = run_command(upload_cmd)
    
    print("Upload complete!")
    print(f"Dataset '{name}' is now available in your Gradient account")
    
    return dataset_id

def main():
    parser = argparse.ArgumentParser(description='Upload dataset to Gradient')
    parser.add_argument('--name', type=str, default='vexus-dataset',
                        help='Name for the dataset in Gradient')
    parser.add_argument('--description', type=str, 
                        default='VEXUS ultrasound dataset',
                        help='Description for the dataset')
    parser.add_argument('--file_path', type=str, 
                        default='/Users/gabe/vexus_dataset.tar.gz',
                        help='Path to the compressed dataset file')
    parser.add_argument('--storage_provider_id', type=str, default='ps_s3',
                        help='Storage provider ID (ps_s3, etc.)')
    
    args = parser.parse_args()
    
    dataset_id = upload_dataset(
        args.name,
        args.description,
        args.file_path,
        args.storage_provider_id
    )
    
    print(f"\nDataset ID: {dataset_id}")
    print("\nYou can use this dataset in your Gradient notebooks and workflows.")

if __name__ == "__main__":
    main() 