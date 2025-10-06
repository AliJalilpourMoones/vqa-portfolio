# Multi-Modal Visual Question Answering (VQA) with ViT and BERT

This repository contains the source code for a Visual Question Answering (VQA) model. The project implements a multi-modal architecture that fuses features from a Vision Transformer (ViT) and a BERT-based text encoder to answer natural language questions about images.

---

## Abstract

Visual Question Answering is a challenging AI task that requires a model to comprehend information from both visual and textual modalities. This project implements a baseline VQA model by fusing the outputs of a pre-trained ViT and a pre-trained BERT. The model is designed to be trained on the VQA v2 dataset and is structured as a classification task over the 1,000 most frequent answers.

---

## 1. Model Architecture

The core of this project is the fusion model that combines features from two powerful uni-modal encoders.



#### ### Vision Encoder (ViT)
* A pre-trained **Vision Transformer (`google/vit-base-patch16-224-in21k`)** processes the input images and outputs a feature vector representing the image's content.

#### ### Text Encoder (BERT)
* A pre-trained **BERT model (`bert-base-uncased`)** processes the input questions and outputs a feature vector representing the question's semantic meaning.

#### ### Fusion Head & Classifier
* The feature vectors from ViT and BERT are **concatenated** and passed through a small feed-forward network (the "fusion head") to allow the features to interact before a final classifier predicts the answer.

---

## 2. Results & Analysis

**[RESULTS PENDING]**
The model will be trained for 5 epochs. The primary evaluation metric will be classification accuracy on the validation set.

#### ### Expected Success Cases
The model is expected to perform well on questions involving:
* Simple Object Recognition and Color Identification.
* Basic Counting of prominent objects.

#### ### Expected Failure Cases (Error Analysis)
The model is expected to struggle with questions that require:
* **Complex Spatial Reasoning:** (e.g., "What is to the left of the tree?")
* **Reading Text in Images (OCR):** (e.g., "What does the sign say?")
* **World Knowledge / Common Sense:** (e.g., "Is this person happy?")

---

## 3. How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AliJalilpourMoones/vqa-portfolio.git](https://github.com/AliJalilpourMoones/vqa-portfolio.git)
    cd vqa-portfolio
    ```
2.  **Download the VQA v2 dataset** as instructed in the `data/README.md` file.

3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the training script:**
    ```bash
    python src/train.py
    ```
