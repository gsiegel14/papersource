# Running on Paperspace Gradient

This guide explains how to run the SDXL fine-tuning project on Paperspace Gradient.

## 1. Create a Gradient Project

1. Sign up or log in to [Paperspace Gradient](https://console.paperspace.com/)
2. Create a new project from the dashboard

## 2. Upload Dataset to Gradient

1. In your Gradient project, go to the "Datasets" tab
2. Create a new dataset
3. Upload your ultrasound images from `/Applications/VEXUS FINAL/VEXUS_Dataset_synthetic`
4. Make sure to maintain the same folder structure with the 9 classes

## 3. Upload Code to Gradient

Either push your code to GitHub and clone it in Gradient, or use the Gradient CLI:

```bash
pip install gradient
gradient login
gradient projects upload --name "sdxl-ultrasound" --path /path/to/your/local/project
```

## 4. Create a Gradient Notebook

1. In your project, go to "Notebooks" and create a new notebook
2. Choose a GPU instance (A4000 or higher recommended for SDXL)
3. Select the PyTorch container (e.g., `Pytorch 2.0.1 with CUDA 11.8`)
4. Connect to your uploaded dataset

## 5. Update Configuration

Once your notebook is running, update the dataset path in the config file:

```python
# Edit configs/training_config.py
dataset_path = "/datasets/your-dataset-name"  # Update this to your Gradient dataset path
```

## 6. Install Dependencies

In your Gradient notebook, install the project dependencies:

```bash
pip install -r requirements.txt
```

## 7. Run Training

Start the training with Weights & Biases tracking:

```bash
# First log in to W&B
wandb login

# Then start training
python src/train/train_sdxl.py --with_tracking
```

## 8. Monitor Training

You can monitor the training progress through:
- Weights & Biases dashboard
- The notebook output

## 9. Generate Images

After training completes, generate images:

```bash
python src/utils/generate_images.py --generate_all --num_images 4
```

## 10. Evaluate Results

Evaluate the quality of your generated images:

```bash
python src/utils/evaluate_model.py --eval_type fid
```

## 11. Download Results

You can download the generated images and trained model from the Gradient UI or using:

```bash
gradient datasets create --name "sdxl-ultrasound-results" --storageProviderId aws
gradient datasets put --id YOUR_DATASET_ID --path models/
gradient datasets put --id YOUR_DATASET_ID --path generated_images/
```

## Tips for Gradient

1. **Persistent Storage**: Save checkpoints frequently using the `--save_interval` parameter
2. **Use Spot Instances**: For cheaper training, use spot instances but make sure to save checkpoints
3. **Memory Optimization**: SDXL requires significant memory, use the provided memory optimizations:
   - LoRA fine-tuning (`use_lora = True` in config)
   - xFormers memory efficient attention (`enable_xformers_memory_efficient_attention = True` in config)
   - Mixed precision training (`mixed_precision = "fp16"` in config) 