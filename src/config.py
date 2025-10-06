# src/config.py

import torch

class Config:
    # --- Project setup ---
    DATA_DIR = "./data/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Model selection ---
    # Using a smaller ViT and BERT for faster training as a baseline
    VISION_MODEL_NAME = "google/vit-base-patch16-224-in21k"
    TEXT_MODEL_NAME = "bert-base-uncased"
    
    # --- Training hyperparameters ---
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    
    # --- Dataset parameters ---
    # In VQA, it's standard to treat it as a classification problem
    # over the N most common answers.
    NUM_TOP_ANSWERS = 1000 
    MAX_TEXT_LENGTH = 32 # Max length for questions
    
    # --- Image processing ---
    IMAGE_SIZE = 224