from transformers import TrainingArguments
import torch

class Config:
    # Model Configuration
    MODEL_NAME = "indolem/indobert-base-uncased"
    MODEL_PATH = "data/models"
    
    # Database Configuration
    MONGODB_URI = "mongodb://localhost:27017"
    DB_NAME = "belajar"
    
    # Check CUDA and set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training Hyperparameters - Optimized for RTX 4050
    EPOCHS = 10
    BATCH_SIZE = 16  # Increased for GPU
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512  # Can use full length with GPU
    
    # Additional Training Parameters
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 1  # Reduced since we have GPU
    
    # Early Stopping Parameters
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.01
    
    # Data Processing
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    NUM_WORKERS = 4  # Increased for better data loading
    
    # Advanced Training Arguments - GPU Optimized
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Can use larger batch size for evaluation
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{MODEL_PATH}/logs",
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,  # Enable mixed precision training
        fp16_opt_level="O1",  # Optimal mixed precision setting
        dataloader_num_workers=NUM_WORKERS,
        group_by_length=True,
        remove_unused_columns=True,
        label_smoothing_factor=0.1,
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        optim="adamw_torch",  # Use PyTorch's AdamW implementation
        half_precision_backend="auto",
    )