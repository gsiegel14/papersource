# SDXL Fine-tuning for Pulse Wave Doppler Ultrasound Images

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sdxl-ultrasound/blob/main/notebooks/train_sdxl_ultrasound.ipynb)

This project provides tools to fine-tune Stable Diffusion XL (SDXL) on pulse wave Doppler ultrasound images. The goal is to generate photorealistic images that are indistinguishable from real ultrasound data by radiologists.

## Dataset

The project is designed to work with the VEXUS dataset that contains ultrasound images in 9 classes:
- Hepatic (Normal, Mild, Severe)
- Portal (Normal, Mild, Severe)
- Renal (Normal, Mild, Severe)

The dataset should be located at: `/Applications/VEXUS FINAL/VEXUS_Dataset_synthetic`

## Project Structure

```
papersource/
├── configs/           # Configuration files
├── models/            # Saved model checkpoints
├── generated_images/  # Output directory for generated images
├── src/
│   ├── data/          # Dataset handling
│   ├── train/         # Training scripts
│   └── utils/         # Utility scripts
└── README.md
```

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the dataset is accessible at the path specified in `configs/training_config.py`.

## Training

To fine-tune SDXL on the ultrasound dataset:

```bash
python src/train/train_sdxl.py --with_tracking
```

### Training Options

- `--output_dir`: Override the default output directory
- `--seed`: Set a random seed for reproducibility (default: 42)
- `--with_tracking`: Enable Weights & Biases tracking
- `--resume_from_checkpoint`: Path to checkpoint to resume training from

The model uses Low-Rank Adaptation (LoRA) for efficient fine-tuning, which dramatically reduces memory requirements and training time.

## Generating Images

After training, you can generate new ultrasound images using:

```bash
python src/utils/generate_images.py --class_type Hepatic_Normal --num_images 4
```

### Generation Options

- `--model_path`: Path to the fine-tuned model
- `--output_dir`: Directory to save generated images
- `--prompt`: Custom prompt for image generation
- `--class_type`: Class type to generate images for (Hepatic_Normal, Portal_Mild, etc.)
- `--num_images`: Number of images to generate
- `--seed`: Random seed for reproducibility
- `--guidance_scale`: Guidance scale for classifier-free guidance
- `--negative_prompt`: Negative prompt for generation
- `--generate_all`: Generate images for all classes

## Configuration

The main configuration file is located at `configs/training_config.py`. You can adjust parameters such as:

- Learning rate
- Batch size
- Training steps
- LoRA parameters
- Image size
- Dataset path

## Example Usage

### Fine-tune the model

```bash
python src/train/train_sdxl.py --with_tracking
```

### Generate images for a specific class

```bash
python src/utils/generate_images.py --class_type Renal_Severe --num_images 8
```

### Generate images for all classes

```bash
python src/utils/generate_images.py --generate_all --num_images 2
```

## Performance Evaluation

To evaluate the quality of generated images and compare them with the original dataset, you can use the following metrics:

1. Fréchet Inception Distance (FID) - to measure the similarity between generated and real images
2. Radiologist evaluation - have radiologists assess the realism of generated images
3. Classification accuracy - train a classifier on real images and test it on generated ones 

## Running on Paperspace Gradient

For high-performance GPU training in the cloud, you can run this project on Paperspace Gradient:

### Quick Start (Automated Setup)

1. Make the setup script executable:
   ```bash
   chmod +x run_on_gradient.sh
   ```

2. Run the setup script:
   ```bash
   ./run_on_gradient.sh
   ```

3. Follow the prompts to configure your training job.

### Manual Setup

For manual setup, follow the instructions in `gradient_setup.md`.

### Using Gradient Workflows

For automated, reproducible training pipelines:

1. Update `gradient_workflow.yaml` with your GitHub repository
2. Create a workflow on Gradient:
   ```bash
   gradient workflows create --name sdxl-ultrasound --specPath gradient_workflow.yaml
   ```

3. Run the workflow:
   ```bash
   gradient workflows run --id YOUR_WORKFLOW_ID \
       --input dataset_id=YOUR_DATASET_ID
   ``` 