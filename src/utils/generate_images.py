import os
import sys
import torch
import argparse
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import numpy as np

sys.path.append(".")
from configs.training_config import SDXLTrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ultrasound images with fine-tuned SDXL")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./models/sdxl-ultrasound-finetuned/final_unet_lora",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./generated_images",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="Custom prompt for image generation, if not specified will use class-based prompts"
    )
    parser.add_argument(
        "--class_type", 
        type=str, 
        choices=["Hepatic_Normal", "Hepatic_Mild", "Hepatic_Severe", 
                 "Portal_Normal", "Portal_Mild", "Portal_Severe",
                 "Renal_Normal", "Renal_Mild", "Renal_Severe"],
        help="Class type to generate images for"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=4,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=7.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--negative_prompt", 
        type=str, 
        default="low quality, blurry, distorted, deformed, artificial, bad contrast",
        help="Negative prompt for generation"
    )
    parser.add_argument(
        "--generate_all", 
        action="store_true",
        help="Generate images for all classes"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = SDXLTrainingConfig()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load base model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.base_model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    
    # Use DPM++ 2M Karras scheduler for better quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        config.base_model_id, 
        subfolder="scheduler",
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    
    # Load LoRA weights if they exist
    if os.path.exists(args.model_path):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.model_path)
        print(f"Loaded LoRA weights from {args.model_path}")
    else:
        print(f"Warning: LoRA weights not found at {args.model_path}")
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        pipe = pipe.to(device)
    
    # Enable memory efficient attention
    pipe.enable_xformers_memory_efficient_attention()
    
    # Function to generate images
    def generate_images(prompt, class_name=None):
        output_subdir = args.output_dir
        if class_name:
            output_subdir = os.path.join(args.output_dir, class_name)
            os.makedirs(output_subdir, exist_ok=True)
            
        print(f"Generating {args.num_images} images for prompt: {prompt}")
        
        for i in range(args.num_images):
            # Generate image
            image = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=30,
                guidance_scale=args.guidance_scale,
            ).images[0]
            
            # Save image
            if class_name:
                output_path = os.path.join(output_subdir, f"{class_name}_{i+1}.png")
            else:
                output_path = os.path.join(output_subdir, f"generated_{i+1}.png")
                
            image.save(output_path)
            print(f"Saved image to {output_path}")
    
    # Generate images
    if args.generate_all:
        for class_name in config.classes:
            prompt = f"A pulse wave Doppler ultrasound image showing {class_name.replace('_', ' ')}"
            generate_images(prompt, class_name)
    elif args.class_type:
        prompt = f"A pulse wave Doppler ultrasound image showing {args.class_type.replace('_', ' ')}"
        generate_images(prompt, args.class_type)
    elif args.prompt:
        generate_images(args.prompt)
    else:
        print("Please specify either --prompt, --class_type, or --generate_all")

if __name__ == "__main__":
    main() 