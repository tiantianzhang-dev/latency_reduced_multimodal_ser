import os
import json
import time
import sys  # âœ… addï¼šcontrol output flush
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, HubertModel, BertTokenizer, BertModel
from transformers import Wav2Vec2ForCTC
from enhanced_ser_model import EnhancedSERModel

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, HubertModel, BertTokenizer, BertModel, Wav2Vec2ForCTC
import logging
from datetime import datetime

# set log with date automatically
log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# output
def log(msg):
    print(msg)
    logging.info(msg)

#  IEMOCAPDatasetã€collate_batchã€EnhancedSERModel 

# âœ… make sure the output is being record into log
sys.stdout.reconfigure(line_buffering=True)

EMO_LABELS = ["happy", "sad", "angry", "neutral"]
LABEL2ID = {e: i for i, e in enumerate(EMO_LABELS)}

def predict_text(asr_processor, asr_model, waveform):
    device = next(asr_model.parameters()).device
    input_values = asr_processor(waveform, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return asr_processor.batch_decode(predicted_ids)[0]

class IEMOCAPDataset(Dataset):
    def __init__(self, json_path, processor, tokenizer, asr_processor, asr_model):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.tokenizer = tokenizer
        self.asr_processor = asr_processor
        self.asr_model = asr_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform, sr = torchaudio.load(item["audio_path"])
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        input_values = self.processor(waveform, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0)
        transcription = predict_text(self.asr_processor, self.asr_model, waveform)
        bert_input = self.tokenizer(transcription, return_tensors="pt", padding=True, truncation=True)
        label = LABEL2ID[item["emotion"]]
        return input_values, bert_input, torch.tensor(label)

def collate_batch(batch):
    audio_tensors, bert_inputs, labels = zip(*batch)
    audio_tensors = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"].squeeze(0) for b in bert_inputs], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"].squeeze(0) for b in bert_inputs], batch_first=True, padding_value=0
    )
    text_input = {"input_ids": input_ids, "attention_mask": attention_mask}

    return audio_tensors, text_input, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    epochs = 100

    os.makedirs("checkpoints", exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
    asr_model.eval()

    train_dataset = IEMOCAPDataset("iemocap_train.json", processor, tokenizer, asr_processor, asr_model)
    val_dataset = IEMOCAPDataset("iemocap_val.json", processor, tokenizer, asr_processor, asr_model)
    test_dataset = IEMOCAPDataset("iemocap_test.json", processor, tokenizer, asr_processor, asr_model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = EnhancedSERModel(hubert_model, bert_model, num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    log(f"ðŸ”§ Training started. Device: {device}")
    log(f"Training set: {len(train_dataset)} samples")
    log(f"Validation set: {len(val_dataset)} samples")
    log(f"Test set: {len(test_dataset)} samples\n")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

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

        # validate
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

        log(f"[Epoch {epoch+1}] "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%, "
            f"Time: {elapsed:.2f}s")

        torch.save(model.state_dict(), f"checkpoints/hubert_epoch_{epoch+1}_valacc_{val_acc:.2f}.pt")

    # test
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
    log(f"\nðŸ”¥ Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()