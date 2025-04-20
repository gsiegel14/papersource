import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

sys.path.append(".")
from configs.training_config import SDXLTrainingConfig
from src.data.dataset import get_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion XL for ultrasound images")
    parser.add_argument("--config", type=str, default="configs/training_config.py", help="Path to config file")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--with_tracking", action="store_true", help="Enable WandB tracking")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    config = SDXLTrainingConfig()
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb" if args.with_tracking else None,
    )
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize weights and biases if tracking is enabled
    if args.with_tracking:
        accelerator.init_trackers(
            project_name="sdxl-ultrasound",
            config={
                "train_batch_size": config.train_batch_size,
                "learning_rate": config.learning_rate,
                "gradient_accumulation_steps": config.gradient_accumulation_steps
            }
        )
    
    # Load tokenizer and text encoder
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        config.base_model_id,
        subfolder="tokenizer",
        revision=None,
    )
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        config.base_model_id,
        subfolder="tokenizer_2",
        revision=None,
    )
    
    text_encoder_1 = CLIPTextModel.from_pretrained(
        config.base_model_id,
        subfolder="text_encoder",
        revision=None,
    )
    
    text_encoder_2 = CLIPTextModel.from_pretrained(
        config.base_model_id,
        subfolder="text_encoder_2",
        revision=None,
    )
    
    vae = AutoencoderKL.from_pretrained(
        config.base_model_id,
        subfolder="vae",
        revision=None,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        config.base_model_id,
        subfolder="unet",
        revision=None,
    )
    
    # Initialize the noise scheduler
    scheduler = DDPMScheduler.from_pretrained(
        config.base_model_id,
        subfolder="scheduler"
    )
    
    # Freeze the VAE and text encoders
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    # Add LoRA to UNet
    if config.use_lora:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
    else:
        # If not using LoRA, we fine-tune the whole UNet
        unet.train()
    
    # Enable memory efficient attention if available
    if config.enable_xformers_memory_efficient_attention and is_xformers_available():
        print("Using xFormers memory efficient attention")
        unet.enable_xformers_memory_efficient_attention()
    
    # Create data loaders
    train_dataloader, val_dataloader, _ = get_data_loaders(config)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_train_steps,
    )
    
    # Prepare models, optimizer, and dataloaders for accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move models to device
    vae.to(accelerator.device)
    text_encoder_1.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    
    # Resume training from checkpoint if specified
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
    
    # Training loop
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    print(f"Running training with batch size {total_batch_size}")
    
    progress_bar = tqdm(range(config.max_train_steps), desc="Training steps")
    global_step = 0
    
    for epoch in range(1):  # We only need one epoch for most fine-tuning
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Get images and prompts
                images = batch["image"].to(accelerator.device)
                prompts = batch["prompt"]
                
                # Get text embeddings
                with torch.no_grad():
                    # Convert images to latent space
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    # Text encoder 1 (CLIP ViT-L/14)
                    text_inputs_1 = tokenizer_1(
                        prompts,
                        padding="max_length",
                        max_length=tokenizer_1.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    text_embeddings_1 = text_encoder_1(
                        text_inputs_1.input_ids
                    )[0]
                    
                    # Text encoder 2 (CLIP ViT-G/14)
                    text_inputs_2 = tokenizer_2(
                        prompts,
                        padding="max_length",
                        max_length=tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    text_embeddings_2 = text_encoder_2(
                        text_inputs_2.input_ids
                    )[0]
                    
                    # Concatenate two text embeddings
                    text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)
                    
                    # Also create an unconditioned embedding for classifier-free guidance training
                    uncond_tokens = [""] * len(prompts)
                    uncond_input_1 = tokenizer_1(
                        uncond_tokens,
                        padding="max_length",
                        max_length=tokenizer_1.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    uncond_embeddings_1 = text_encoder_1(
                        uncond_input_1.input_ids
                    )[0]
                    
                    uncond_input_2 = tokenizer_2(
                        uncond_tokens,
                        padding="max_length",
                        max_length=tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    uncond_embeddings_2 = text_encoder_2(
                        uncond_input_2.input_ids
                    )[0]
                    
                    uncond_embeddings = torch.cat([uncond_embeddings_1, uncond_embeddings_2], dim=-1)
                    
                    # Concatenate the conditioned and unconditioned text embeddings for CFG
                    all_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, 1000, (batch_size,), device=accelerator.device
                ).long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_embeddings
                
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                
                # Get the target for loss
                target = noise
                
                loss = F.mse_loss(model_pred, target, reduction="mean")
                
                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                
                # Update optimizer and learning rate
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if we should save and log
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % config.save_interval == 0:
                    if accelerator.is_main_process:
                        # Save checkpoint
                        checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(checkpoint_dir)
                        
                        # Save PEFT/LoRA adapters
                        if config.use_lora:
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            unwrapped_unet.save_pretrained(os.path.join(checkpoint_dir, "unet_lora"))
                
                # Log metrics
                if args.with_tracking:
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "step": global_step,
                        },
                        step=global_step,
                    )
                
                train_loss = 0.0
                
            # Break if max steps reached
            if global_step >= config.max_train_steps:
                break
    
    # Save the final model
    if accelerator.is_main_process:
        # Save PEFT/LoRA adapters
        if config.use_lora:
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(os.path.join(config.output_dir, "final_unet_lora"))
    
    # End wandb run
    if args.with_tracking:
        accelerator.end_training()

if __name__ == "__main__":
    main() 