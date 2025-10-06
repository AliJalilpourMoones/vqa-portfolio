# src/model.py

import torch
import torch.nn as nn
from transformers import ViTModel, BertModel

class VqaModel(nn.Module):
    def __init__(self, vision_model_name, text_model_name, num_classes):
        super().__init__()
        # --- Vision Encoder ---
        self.vit = ViTModel.from_pretrained(vision_model_name)
        
        # --- Text Encoder ---
        self.bert = BertModel.from_pretrained(text_model_name)
        
        # --- Fusion Head and Classifier ---
        # The input dimension is the sum of the [CLS] token dimensions from both models
        fusion_dim = self.vit.config.hidden_size + self.bert.config.hidden_size
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # --- Get Image Features ---
        # We only care about the [CLS] token representation
        image_features = self.vit(pixel_values=pixel_values).pooler_output
        
        # --- Get Text Features ---
        # We only care about the [CLS] token representation
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        
        # --- Fuse Features ---
        combined_features = torch.cat((image_features, text_features), dim=1)
        fused_output = self.fusion_head(combined_features)
        
        # --- Classify ---
        logits = self.classifier(fused_output)
        
        # --- Calculate Loss (if labels are provided) ---
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits}