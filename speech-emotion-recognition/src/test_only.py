import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, HubertModel, BertTokenizer, BertModel, Wav2Vec2ForCTC
from enhanced_ser_model import EnhancedSERModel
from HUBERT import IEMOCAPDataset, collate_batch  # ðŸ‘ˆ replace to real data name

def test_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and processors
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
    asr_model.eval()

    # Load test set
    test_dataset = IEMOCAPDataset("iemocap_test.json", processor, tokenizer, asr_processor, asr_model)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_batch)

    # Load model + weights
    model = EnhancedSERModel(hubert_model, bert_model, num_classes=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for audio_input, text_input, labels in test_loader:
            audio_input = audio_input.to(device)
            text_input = {k: v.to(device) for k, v in text_input.items()}
            labels = labels.to(device)

            outputs = model(audio_input, text_input)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"\nðŸ§ª Test Accuracy from {checkpoint_path}: {acc:.2f}%")

# Optional: test directly
if __name__ == "__main__":
    test_model("checkpoints/hubert_epoch_11_valacc_74.37.pt") 
