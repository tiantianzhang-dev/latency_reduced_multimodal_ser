# Setup Guide

This guide provides detailed instructions for setting up the Speech Emotion Recognition project.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS 10.14+, or Windows 10+ with WSL
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
  - Minimum: 8 GB VRAM
  - Recommended: 16+ GB VRAM
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 20 GB free space (for dataset and models)

### Software Dependencies

- CUDA Toolkit 11.0+ (for GPU support)
- cuDNN 8.0+ (for GPU support)
- Git
- pip or conda

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Create Virtual Environment

#### Using venv (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using conda

```bash
conda create -n ser python=3.8
conda activate ser
```

### 3. Install PyTorch

Visit [pytorch.org](https://pytorch.org) and install the appropriate version for your system.

#### For CUDA 11.8 (example)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU only

```bash
pip install torch torchvision torchaudio
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Dataset Setup

### Obtaining IEMOCAP

1. Visit the IEMOCAP website: https://sail.usc.edu/iemocap/
2. Fill out the request form with your institutional email
3. Wait for approval (typically 1-2 weeks)
4. Download the dataset after approval

### Organizing the Dataset

1. Extract the IEMOCAP archive
2. Place it in the `data/` directory:

```bash
mkdir -p data/IEMOCAP
# Extract downloaded archive to data/IEMOCAP/
```

Expected structure:
```
data/
└── IEMOCAP/
    ├── Session1/
    ├── Session2/
    ├── Session3/
    ├── Session4/
    └── Session5/
```

### Preprocessing the Dataset

1. Process the raw IEMOCAP data:

```bash
cd src
python process.py
```

This creates `iemocap_all.json` in the `data/` directory.

2. Split into train/validation/test sets:

```bash
python split_iemocap.py
```

This creates:
- `iemocap_train.json`
- `iemocap_val.json`
- `iemocap_test.json`

## Configuration

### Model Configuration

Edit `src/train.py` to adjust hyperparameters:

```python
batch_size = 2       # Adjust based on available GPU memory
epochs = 100         # Number of training epochs
learning_rate = 1e-5 # Learning rate for optimizer
```

### Training Configuration for SLURM

Edit `scripts/submit_hubert.sh` for cluster training:

```bash
#SBATCH --gres=gpu:1          # Number of GPUs
#SBATCH --mem=32G             # Memory allocation
#SBATCH --time=24:00:00       # Time limit
```

## Pre-trained Models

The following models are automatically downloaded on first use:

1. **HuBERT Large**: `facebook/hubert-large-ls960-ft` (~1.2 GB)
2. **BERT Base**: `bert-base-uncased` (~440 MB)
3. **Wav2vec 2.0**: `facebook/wav2vec2-large-960h` (~1.2 GB)

Total download size: ~3 GB

Models are cached in `~/.cache/huggingface/transformers/`

## Training

### Local Training

```bash
cd src
python train.py
```

or

```bash
python HUBERT.py  # Legacy training script
```

### Cluster Training (SLURM)

```bash
sbatch scripts/submit_hubert.sh
```

Monitor job:
```bash
squeue -u $USER
tail -f training_log_*.txt
```

### Training Outputs

- **Checkpoints**: Saved in `checkpoints/` directory
  - Format: `hubert_epoch_{N}_valacc_{XX.XX}.pt`
- **Logs**: `training_log_YYYYMMDD_HHMMSS.txt`

## Testing

### Run Inference

```bash
cd src
python test_only.py
```

### Load Checkpoint

```python
import torch
from enhanced_ser_model import EnhancedSERModel

# Load model
checkpoint = torch.load('checkpoints/hubert_epoch_2_valacc_74.19.pt')
model.load_state_dict(checkpoint)
model.eval()
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

**Solution**: Reduce batch size
```python
batch_size = 1  # in train.py
```

#### Slow Training

**Possible causes**:
- CPU-only mode (check CUDA availability)
- Large batch size
- Slow disk I/O

**Solutions**:
- Verify GPU is being used: `nvidia-smi`
- Reduce batch size
- Move data to faster storage (SSD)

#### Model Download Fails

**Solution**: Manually download and cache models
```bash
python -c "from transformers import HubertModel; HubertModel.from_pretrained('facebook/hubert-large-ls960-ft')"
```

#### Import Errors

**Solution**: Verify all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

### Getting Help

1. Check existing issues: https://github.com/yourusername/speech-emotion-recognition/issues
2. Open a new issue with:
   - Error message
   - System information
   - Steps to reproduce
3. Contact the authors

## Performance Optimization

### Mixed Precision Training

Enable FP16 training for faster computation:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    outputs = model(audio_input, text_input)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
accumulation_steps = 4

for i, (audio, text, labels) in enumerate(train_loader):
    outputs = model(audio, text)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### DataLoader Optimization

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster data transfer to GPU
    collate_fn=collate_batch
)
```

## Next Steps

1. Train the baseline model
2. Evaluate on test set
3. Experiment with hyperparameters
4. Implement the proposed simplified architecture
5. Add data augmentation and regularization

For more details, see:
- [README.md](README.md) - Project overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [docs/paper.pdf](docs/paper.pdf) - Research paper
