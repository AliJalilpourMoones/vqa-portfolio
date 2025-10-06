# src/train.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
from tqdm import tqdm

# Import from our other project files
from config import Config
from dataset import VqaDataset, prepare_data
from model import VqaModel

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        # Move batch to device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs["loss"]
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return total_loss / len(data_loader)

def main():
    config = Config()
    
    # --- Prepare Data ---
    # This step is simplified. In a real project, this would be a complex script.
    train_df, val_df, num_classes = prepare_data(config.DATA_DIR)
    
    # --- Initialize Tokenizer and Image Processor ---
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(config.VISION_MODEL_NAME)

    # --- Create Datasets and DataLoaders ---
    print("Creating datasets...")
    train_dataset = VqaDataset(train_df, os.path.join(config.DATA_DIR, "train2014"), tokenizer, image_processor)
    # val_dataset = VqaDataset(val_df, os.path.join(config.DATA_DIR, "val2014"), tokenizer, image_processor) # Validation would be similar
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # --- Initialize Model, Optimizer, and Loss ---
    print("Initializing model...")
    model = VqaModel(
        vision_model_name=config.VISION_MODEL_NAME,
        text_model_name=config.TEXT_MODEL_NAME,
        num_classes=num_classes
    ).to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, config.DEVICE)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # In a real scenario, you would run an evaluation loop here on the validation set
        # and save the best model.
    
    print("\nTraining complete.")

if __name__ == "__main__":
    main()