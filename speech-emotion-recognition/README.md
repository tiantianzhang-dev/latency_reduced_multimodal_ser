# Single-Layered Fusion in Uni-Task Multimodal Speech Emotion Recognition Models

This repository contains the implementation of the paper "Single-Layered Fusion in Uni-Task Multimodal Speech Emotion Recognition Models" by Xuefei Bian, Hao-wei Liang, and Tiantian Zhang.

## Abstract

We propose a single-layered fusion uni-tasking Speech Emotion Recognition (SER) multimodal architecture that reduces training and inference time by 43% and 38%, respectively, compared to a similar multi-layer fusion model, while achieving state-of-the-art accuracy (74.9% on IEMOCAP dataset). The fusion mechanism dynamically integrates acoustic and semantic features using a temporal-aware sigmoid gate.

## Key Features

- **Efficient Architecture**: Single-layer fusion gate for optimal computational cost/accuracy trade-off
- **Multimodal Integration**: Combines acoustic (HuBERT) and semantic (BERT) features
- **State-of-the-art Performance**: 74.9% weighted accuracy on IEMOCAP dataset
- **Real-time Ready**: 38% faster inference compared to multi-layer fusion models

## Repository Structure

```
speech-emotion-recognition/
├── src/
│   ├── HUBERT.py                 # Main training script
│   ├── enhanced_ser_model.py     # Model architecture (CmGI baseline)
│   ├── process.py                # Data processing utilities
│   ├── split_iemocap.py          # Dataset splitting script
│   └── test_only.py              # Testing script
├── data/
│   └── README.md                 # Instructions for dataset preparation
├── models/
│   └── README.md                 # Pre-trained model information
├── checkpoints/
│   └── .gitkeep                  # Checkpoint storage directory
├── docs/
│   └── paper.pdf                 # Original research paper
├── submit_hubert.sh              # SLURM job submission script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the IEMOCAP dataset (requires permission from USC):
   - Visit: https://sail.usc.edu/iemocap/
   - Follow instructions in `data/README.md`

## Dataset Preparation

### IEMOCAP Dataset

The IEMOCAP dataset contains approximately 12 hours of multimodal data including audio, video, and textual transcriptions from ten American English speakers.

1. After obtaining the dataset, place it in the `data/` directory
2. Run the preprocessing script:
```bash
python src/process.py
```

3. Split the dataset into train/validation/test sets:
```bash
python src/split_iemocap.py
```

This creates three JSON files:
- `iemocap_train.json`: Training set (Sessions 1, 3, 4, 5)
- `iemocap_val.json`: Validation set (20% of training sessions)
- `iemocap_test.json`: Test set (Session 2)

### Dataset Statistics

- **Training samples**: ~2,230
- **Validation samples**: ~558
- **Test samples**: ~678
- **Emotion classes**: 4 (happy, sad, angry, neutral)
- **Note**: "Excited" labels are merged with "happy"

## Model Architecture

### Components

1. **Acoustic Feature Extraction**
   - Model: HuBERT Large (facebook/hubert-large-ls960-ft)
   - Output: 1024-dimensional acoustic representations

2. **Semantic Feature Extraction**
   - ASR: Wav2vec 2.0 (facebook/wav2vec2-large-960h)
   - Text Encoder: BERT (bert-base-uncased)
   - Output: 1024-dimensional semantic representations (projected from 768)

3. **Fusion Mechanism**
   - Cross-modal attention with temporal-aware gated fusion
   - Formula: `G = σ(W·concat([X, C]) + b)`
   - Output: `fused = X × G + C × (1 - G)`

4. **Classification**
   - Fully connected layer mapping 2048-dimensional features to 4 emotion classes

### Current Implementation Note

The code currently implements the **CmGI baseline model** (multi-layer fusion) from Gao et al. [18]. The proposed simplified single-layer architecture would require modifications to remove separate attention projections.

## Training

### Local Training

```bash
python src/HUBERT.py
```

### Cluster Training (SLURM)

```bash
sbatch submit_hubert.sh
```

### Training Configuration

- **Batch size**: 2
- **Learning rate**: 1e-5
- **Optimizer**: Adam
- **Loss function**: Cross-entropy
- **Epochs**: 100 (with checkpoint saving per epoch)

### Training Results

Training logs show:
- Epoch 1: 57.89% train accuracy, 70.43% validation accuracy
- Epoch 2: 80.18% train accuracy, 74.19% validation accuracy
- Epoch 6: 95.43% train accuracy, 73.30% validation accuracy (overfitting begins)

## Testing

Run inference on the test set:

```bash
python src/test_only.py
```

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Weighted Accuracy (WA) | 74.9% |
| Unweighted Accuracy (UA) | - |
| Training Time Reduction | 43% vs multi-layer |
| Inference Time Reduction | 38% vs multi-layer |

### Comparison with State-of-the-art

| Model | Type | Fusion | WA on IEMOCAP |
|-------|------|--------|---------------|
| CmGI (Gao et al.) | Uni-task | Multi-layer | 79.5% |
| MMER (Ghosh et al.) | Multi-task | Complex | 81.2% |
| Yu et al. | Uni-task | Single-layer | 69.65% |
| **Proposed** | Uni-task | Single-layer | **74.9%** |

## Known Issues and Limitations

### Overfitting

The model shows clear overfitting patterns:
- Training accuracy reaches 95%+ by epoch 6
- Validation loss increases from epoch 3 onwards
- Validation accuracy plateaus at ~73-75%

### Causes

1. **Limited dataset**: IEMOCAP has only 6,412 utterances (~12 hours)
2. **No regularization**: Missing dropout, weight decay, early stopping
3. **No data augmentation**: No noise injection, time stretching, etc.

### Recommendations

- Implement dropout layers (0.3-0.5)
- Add weight decay to optimizer
- Apply early stopping based on validation loss
- Use data augmentation techniques
- Consider cross-validation instead of single train/val/test split

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@article{bian2024single,
  title={Single-Layered Fusion in Uni-Task Multimodal Speech Emotion Recognition Models},
  author={Bian, Xuefei and Liang, Hao-wei and Zhang, Tiantian},
  journal={Term Paper, Speech Recognition II},
  year={2024},
  institution={University of Groningen, Campus Fryslân}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IEMOCAP dataset provided by USC
- Pre-trained models from Hugging Face Transformers
- Reference implementations from Gao et al. (CmGI) and Ghosh et al. (MMER)

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## References

Key references:
- [Gao et al. 2024] Speech Emotion Recognition with Multi-level Acoustic and Semantic Information Extraction and Interaction
- [Ghosh et al. 2022] MMER: Multimodal Multi-task Learning for Speech Emotion Recognition
- [Hsu et al. 2021] HuBERT: Self-supervised Speech Representation Learning
- [Baevski et al. 2020] wav2vec 2.0: A Framework for Self-supervised Learning
- [Devlin et al. 2019] BERT: Pre-training of Deep Bidirectional Transformers

For complete references, see the paper in `docs/` directory.
