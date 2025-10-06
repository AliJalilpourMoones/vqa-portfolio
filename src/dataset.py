# src/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from collections import Counter

class VqaDataset(Dataset):
    def __init__(self, df, image_dir, tokenizer, image_processor, is_train=True):
        self.df = df
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- Image ---
        image_path = os.path.join(self.image_dir, row['image_id'] + ".jpg")
        image = Image.open(image_path).convert("RGB")
        processed_image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze()

        # --- Question ---
        question = row['question']
        tokenized_question = self.tokenizer(
            question,
            padding='max_length',
            max_length=32, # From Config
            truncation=True,
            return_tensors="pt"
        )
        
        # --- Answer ---
        if self.is_train:
            answer_label = row['answer_label']
            return {
                "pixel_values": processed_image,
                "input_ids": tokenized_question['input_ids'].squeeze(),
                "attention_mask": tokenized_question['attention_mask'].squeeze(),
                "labels": torch.tensor(answer_label, dtype=torch.long)
            }
        else: # For validation or test, we don't always have labels
             return {
                "pixel_values": processed_image,
                "input_ids": tokenized_question['input_ids'].squeeze(),
                "attention_mask": tokenized_question['attention_mask'].squeeze(),
            }

# --- Data Preprocessing Helper Function ---
def prepare_data(data_dir):
    # This is a simplified preprocessing pipeline. A real one would be more complex.
    # It reads the JSON files, merges them, and creates labels for the top N answers.
    
    print("Preparing data... This may take a moment.")
    
    # This function would contain logic to:
    # 1. Load the train/val questions and annotations JSON files.
    # 2. Merge them based on question_id and image_id.
    # 3. For the training set, find the 1000 most common answers.
    # 4. Create a mapping from these answers to integer labels (0-999).
    # 5. Create a final DataFrame with columns: 'image_id', 'question', 'answer', 'answer_label'.
    # 6. For simplicity here, we'll create dummy DataFrames.
    #    In a real scenario, you'd replace this with actual data loading.
    
    print("WARNING: Using dummy data. Replace with real VQA data loading.")
    dummy_data = {
        'image_id': [f'COCO_train2014_{i:012d}' for i in range(100)],
        'question': ["What color is the cat?"] * 100,
        'answer': ["brown"] * 100,
        'answer_label': [1] * 100 # Assuming 'brown' is label 1
    }
    train_df = pd.DataFrame(dummy_data)
    val_df = pd.DataFrame(dummy_data)
    
    # The number of classes for our model is the number of top answers
    num_classes = 1000 # From Config
    
    return train_df, val_df, num_classes