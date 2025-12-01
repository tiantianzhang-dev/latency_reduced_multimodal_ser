# Pre-trained Models

This directory contains information about pre-trained models used in the project.

## Model Components

### 1. Acoustic Feature Extractor: HuBERT

**Model**: `facebook/hubert-large-ls960-ft`

- **Type**: Self-supervised speech representation model
- **Parameters**: ~300M
- **Training data**: LibriSpeech 960 hours
- **Output dimension**: 1024
- **Layers**: 24 transformer layers
- **Download**: Automatically downloaded via Hugging Face Transformers

```python
from transformers import Wav2Vec2Processor, HubertModel

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
```

**Citation**:
```bibtex
@article{hsu2021hubert,
  title={HuBERT: Self-supervised speech representation learning by masked prediction of hidden units},
  author={Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={3451--3460},
  year={2021}
}
```

### 2. Automatic Speech Recognition: Wav2vec 2.0

**Model**: `facebook/wav2vec2-large-960h`

- **Type**: ASR model based on wav2vec 2.0
- **Parameters**: ~300M
- **Training data**: LibriSpeech 960 hours
- **Purpose**: Generate text transcriptions from audio
- **Download**: Automatically downloaded via Hugging Face Transformers

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
```

**Citation**:
```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={12449--12460},
  year={2020}
}
```

### 3. Semantic Feature Extractor: BERT

**Model**: `bert-base-uncased`

- **Type**: Bidirectional transformer language model
- **Parameters**: ~110M
- **Training data**: BooksCorpus and English Wikipedia
- **Output dimension**: 768 (projected to 1024)
- **Layers**: 12 transformer layers
- **Download**: Automatically downloaded via Hugging Face Transformers

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

**Citation**:
```bibtex
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics},
  pages={4171--4186},
  year={2019}
}
```

## Fine-tuned Checkpoints

After training, model checkpoints are saved in `../checkpoints/` with the naming format:

```
hubert_epoch_{epoch}_valacc_{val_acc}.pt
```

Example:
- `hubert_epoch_2_valacc_74.19.pt`
- `hubert_epoch_3_valacc_75.27.pt`

### Loading a Checkpoint

```python
from src.enhanced_ser_model import EnhancedSERModel
from transformers import HubertModel, BertModel

# Initialize models
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Create SER model
model = EnhancedSERModel(hubert_model, bert_model, num_classes=4)

# Load checkpoint
checkpoint = torch.load("../checkpoints/hubert_epoch_2_valacc_74.19.pt")
model.load_state_dict(checkpoint)
model.eval()
```

## Model Size and Requirements

| Component | Parameters | Memory (FP32) | Memory (FP16) |
|-----------|-----------|---------------|---------------|
| HuBERT Large | ~300M | ~1.2 GB | ~600 MB |
| BERT Base | ~110M | ~440 MB | ~220 MB |
| Wav2vec 2.0 | ~300M | ~1.2 GB | ~600 MB |
| Fusion Module | ~2M | ~8 MB | ~4 MB |
| **Total** | ~712M | ~2.8 GB | ~1.4 GB |

### GPU Requirements

- **Minimum**: 8 GB VRAM (batch size 1-2)
- **Recommended**: 16 GB VRAM (batch size 4-8)
- **Optimal**: 24+ GB VRAM (batch size 16+)

## Download Instructions

All models are automatically downloaded when first used. Ensure you have:

1. Internet connection for initial download
2. Sufficient disk space (~3-4 GB for all models)
3. Hugging Face Transformers library installed

Models are cached in:
- Linux/Mac: `~/.cache/huggingface/transformers/`
- Windows: `C:\Users\{username}\.cache\huggingface\transformers\`

## Troubleshooting

### Slow Download
If downloads are slow, you can manually download and place models in the cache directory.

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training (FP16)
- Use model parallelism for multi-GPU setups

### Model Not Found
Ensure you have the latest version of transformers:
```bash
pip install --upgrade transformers
```
