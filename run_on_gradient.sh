#!/bin/bash

# Check if gradient CLI is installed
if ! command -v gradient &> /dev/null; then
    echo "Installing Gradient CLI..."
    pip install gradient
fi

# Login to Gradient if needed
if ! gradient auth list | grep -q "active"; then
    echo "Please log in to Gradient:"
    gradient login
fi

# Prompt for GitHub repository (if using GitHub)
read -p "Enter your GitHub repository URL (leave empty if uploading local files): " GITHUB_REPO

# Prompt for dataset name
read -p "Enter a name for your dataset on Gradient: " DATASET_NAME

# Prompt for dataset path
read -p "Enter the path to your ultrasound dataset [default: /Applications/VEXUS FINAL/VEXUS_Dataset_synthetic]: " DATASET_PATH
DATASET_PATH=${DATASET_PATH:-"/Applications/VEXUS FINAL/VEXUS_Dataset_synthetic"}

# Prompt for machine type
echo "Select a machine type for training:"
echo "1) A4000 (16GB VRAM)"
echo "2) A5000 (24GB VRAM)"
echo "3) A6000 (48GB VRAM)"
read -p "Enter your choice [1-3, default: 1]: " MACHINE_CHOICE
case $MACHINE_CHOICE in
    2) MACHINE_TYPE="A5000" ;;
    3) MACHINE_TYPE="A6000" ;;
    *) MACHINE_TYPE="A4000" ;;
esac

# Prompt for training settings
read -p "Number of training steps [default: 2000]: " TRAINING_STEPS
TRAINING_STEPS=${TRAINING_STEPS:-2000}

read -p "Save checkpoint interval [default: 200]: " SAVE_INTERVAL
SAVE_INTERVAL=${SAVE_INTERVAL:-200}

read -p "Use Weights & Biases tracking? [y/N]: " USE_WANDB
USE_WANDB=${USE_WANDB:-n}

# Create a project on Gradient
echo "Creating Gradient project..."
PROJECT_ID=$(gradient projects create --name "sdxl-ultrasound" --readme "SDXL fine-tuning for ultrasound images" | grep -oP 'id: \K[a-z0-9]+')
echo "Created project with ID: $PROJECT_ID"

# Create and upload dataset
echo "Creating dataset on Gradient..."
DATASET_ID=$(gradient datasets create --name "$DATASET_NAME" --storageProviderId aws | grep -oP 'id: \K[a-z0-9]+')
echo "Created dataset with ID: $DATASET_ID"

echo "Uploading dataset to Gradient..."
gradient datasets put --id $DATASET_ID --path "$DATASET_PATH"

# Upload code or use GitHub
if [ -z "$GITHUB_REPO" ]; then
    echo "Uploading code to Gradient..."
    gradient projects upload --id $PROJECT_ID --path .
    USE_GITHUB="false"
else
    USE_GITHUB="true"
fi

# Create a notebook
echo "Creating Notebook on Gradient..."
NOTEBOOK_PARAMS=""
if [ "$USE_GITHUB" = "true" ]; then
    NOTEBOOK_PARAMS="--container custom --containerUrl $GITHUB_REPO"
else
    NOTEBOOK_PARAMS="--container custom --containerUrl paperspace/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
fi

NOTEBOOK_ID=$(gradient notebooks create --projectId $PROJECT_ID --machineType $MACHINE_TYPE $NOTEBOOK_PARAMS | grep -oP 'id: \K[a-z0-9]+')
echo "Created notebook with ID: $NOTEBOOK_ID"

# Attach dataset to notebook
echo "Attaching dataset to notebook..."
gradient notebooks datasetAttach --id $NOTEBOOK_ID --datasetId $DATASET_ID

# Generate startup script
cat << EOF > startup.sh
#!/bin/bash
cd /notebooks

# Clone repository if using GitHub
if [ "$USE_GITHUB" = "true" ]; then
    git clone $GITHUB_REPO /notebooks/sdxl-ultrasound
    cd /notebooks/sdxl-ultrasound
fi

# Install dependencies
pip install -r requirements.txt

# Update configuration
sed -i "s|dataset_path = .*|dataset_path = \"/datasets/$DATASET_ID\"|g" configs/training_config.py
sed -i "s|max_train_steps = .*|max_train_steps = $TRAINING_STEPS|g" configs/training_config.py
sed -i "s|save_interval = .*|save_interval = $SAVE_INTERVAL|g" configs/training_config.py

# Start training
if [ "$USE_WANDB" = "y" ] || [ "$USE_WANDB" = "Y" ]; then
    python src/train/train_sdxl.py --with_tracking
else
    python src/train/train_sdxl.py
fi

# Generate images after training
python src/utils/generate_images.py --generate_all --num_images 4

# Evaluate model
python src/utils/evaluate_model.py --eval_type fid --real_data_dir /datasets/$DATASET_ID
EOF

chmod +x startup.sh

# Upload startup script
echo "Uploading startup script..."
gradient notebooks startupScriptUpload --id $NOTEBOOK_ID --path startup.sh

# Start the notebook
echo "Starting notebook..."
gradient notebooks start --id $NOTEBOOK_ID

echo "Setup complete!"
echo "Your SDXL fine-tuning is running on Gradient."
echo "You can monitor progress at: https://console.paperspace.com/projects/$PROJECT_ID/notebooks/$NOTEBOOK_ID"
echo ""
echo "After training completes, you can download your model and generated images from the Gradient console." 