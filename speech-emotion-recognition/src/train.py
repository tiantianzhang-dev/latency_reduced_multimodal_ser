"""
Main training script for Speech Emotion Recognition (SER) model.

This script implements the training pipeline for a multimodal SER model that combines
acoustic features (HuBERT) and semantic features (BERT) using a cross-modal gated
interaction fusion mechanism.

Author: Xuefei Bian, Hao-wei Liang, Tiantian Zhang
Date: 2024
"""

import os
import json
import time
import sys
import logging
from datetime import datetime

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    HubertModel,
    BertTokenizer,
    BertModel,
    Wav2Vec2ForCTC
)

from enhanced_ser_model import EnhancedSERModel

# Configure logging
log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Ensure output is flushed to log immediately
sys.stdout.reconfigure(line_buffering=True)

# Emotion labels
EMO_LABELS = ["happy", "sad", "angry", "neutral"]
LABEL2ID = {e: i for i, e in enumerate(EMO_LABELS)}


def log(msg):
    """Print message to console and log file."""
    print(msg)
    logging.info(msg)


def predict_text(asr_processor, asr_model, waveform):
    """
    Use ASR model to predict text transcription from audio waveform.
    
    Args:
        asr_processor: Wav2Vec2 processor for audio preprocessing
        asr_model: Wav2Vec2ForCTC model for speech recognition
        waveform: Audio waveform tensor
        
    Returns:
        str: Predicted text transcription
    """
    device = next(asr_model.parameters()).device
    input_values = asr_processor(
        waveform, 
        return_tensors="pt", 
        sampling_rate=16000
    ).input_values.to(device)
    
    with torch.no_grad():
        logits = asr_model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    return asr_processor.batch_decode(predicted_ids)[0]


class IEMOCAPDataset(Dataset):
    """
    IEMOCAP dataset loader for speech emotion recognition.
    
    This dataset loads audio files, generates text transcriptions using ASR,
    and returns preprocessed features for both acoustic and semantic modalities.
    """
    
    def __init__(self, json_path, processor, tokenizer, asr_processor, asr_model):
        """
        Initialize IEMOCAP dataset.
        
        Args:
            json_path: Path to JSON file containing dataset information
            processor: HuBERT processor for acoustic features
            tokenizer: BERT tokenizer for text features
            asr_processor: Wav2Vec2 processor for ASR
            asr_model: Wav2Vec2 model for generating transcriptions
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.tokenizer = tokenizer
        self.asr_processor = asr_processor
        self.asr_model = asr_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            tuple: (audio_input, text_input, label)
        """
        item = self.data[idx]
        
        # Load and preprocess audio
        waveform, sr = torchaudio.load(item["audio_path"])
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        # Convert stereo to mono if necessary
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Process audio for HuBERT
        input_values = self.processor(
            waveform, 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_values.squeeze(0)
        
        # Generate transcription using ASR
        transcription = predict_text(self.asr_processor, self.asr_model, waveform)
        
        # Tokenize text for BERT
        bert_input = self.tokenizer(
            transcription, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Get emotion label
        label = LABEL2ID[item["emotion"]]
        
        return input_values, bert_input, torch.tensor(label)


def collate_batch(batch):
    """
    Collate function for batching variable-length samples.
    
    Args:
        batch: List of (audio, text, label) tuples
        
    Returns:
        tuple: (padded_audio, padded_text, labels)
    """
    audio_tensors, bert_inputs, labels = zip(*batch)
    
    # Pad audio tensors
    audio_tensors = torch.nn.utils.rnn.pad_sequence(
        audio_tensors, 
        batch_first=True, 
        padding_value=0
    )
    
    # Stack labels
    labels = torch.tensor(labels)

    # Pad text inputs
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"].squeeze(0) for b in bert_inputs], 
        batch_first=True, 
        padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"].squeeze(0) for b in bert_inputs], 
        batch_first=True, 
        padding_value=0
    )
    text_input = {"input_ids": input_ids, "attention_mask": attention_mask}

    return audio_tensors, text_input, labels


def main():
    """Main training loop."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    epochs = 100

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Load pre-trained models
    log("Loading pre-trained models...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    # Load ASR model
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
    asr_model.eval()

    # Load datasets
    log("Loading datasets...")
    train_dataset = IEMOCAPDataset(
        "iemocap_train.json", processor, tokenizer, asr_processor, asr_model
    )
    val_dataset = IEMOCAPDataset(
        "iemocap_val.json", processor, tokenizer, asr_processor, asr_model
    )
    test_dataset = IEMOCAPDataset(
        "iemocap_test.json", processor, tokenizer, asr_processor, asr_model
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    # Initialize model
    model = EnhancedSERModel(hubert_model, bert_model, num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Training info
    log(f"Training started. Device: {device}")
    log(f"Training set: {len(train_dataset)} samples")
    log(f"Validation set: {len(val_dataset)} samples")
    log(f"Test set: {len(test_dataset)} samples\n")

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Train for one epoch
        for audio_input, text_input, labels in train_loader:
            audio_input = audio_input.to(device)
            text_input = {k: v.to(device) for k, v in text_input.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio_input, text_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for audio_input, text_input, labels in val_loader:
                audio_input = audio_input.to(device)
                text_input = {k: v.to(device) for k, v in text_input.items()}
                labels = labels.to(device)

                outputs = model(audio_input, text_input)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        elapsed = time.time() - start_time

        # Log results
        log(
            f"[Epoch {epoch+1}] "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%, "
            f"Time: {elapsed:.2f}s"
        )

        # Save checkpoint
        torch.save(
            model.state_dict(), 
            f"checkpoints/hubert_epoch_{epoch+1}_valacc_{val_acc:.2f}.pt"
        )

    # Final evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for audio_input, text_input, labels in test_loader:
            audio_input = audio_input.to(device)
            text_input = {k: v.to(device) for k, v in text_input.items()}
            labels = labels.to(device)

            outputs = model(audio_input, text_input)
            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100 * test_correct / test_total
    log(f"\nFinal Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
