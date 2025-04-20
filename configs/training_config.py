"""
Configuration file for SDXL fine-tuning on pulse wave Doppler ultrasound images
"""

class SDXLTrainingConfig:
    # Base model
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Dataset parameters
    dataset_path = "/Applications/VEXUS FINAL/VEXUS_Dataset_synthetic"
    train_split_ratio = 0.7
    validation_split_ratio = 0.15
    test_split_ratio = 0.15
    image_size = 1024  # SDXL native resolution
    
    # Classes (derived from dataset folder names)
    classes = [
        "Hepatic_Normal", "Hepatic_Mild", "Hepatic_Severe",
        "Portal_Normal", "Portal_Mild", "Portal_Severe",
        "Renal_Normal", "Renal_Mild", "Renal_Severe"
    ]
    
    # Training parameters
    train_batch_size = 1
    gradient_accumulation_steps = 4
    mixed_precision = "fp16"
    learning_rate = 1e-5
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    max_train_steps = 2000
    save_interval = 200
    
    # LoRA parameters
    use_lora = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    # Output paths
    output_dir = "./models/sdxl-ultrasound-finetuned"
    logging_dir = "./logs"
    
    # Hardware config
    enable_xformers_memory_efficient_attention = True 