# Project Structure

This document describes the organization of the Speech Emotion Recognition repository.

## Directory Tree

```
speech-emotion-recognition/
├── .gitignore                      # Git ignore patterns
├── LICENSE                         # MIT License
├── README.md                       # Project overview and documentation
├── SETUP.md                        # Detailed setup instructions
├── CONTRIBUTING.md                 # Contribution guidelines
├── requirements.txt                # Python dependencies
│
├── src/                            # Source code
│   ├── train.py                    # Main training script (cleaned)
│   ├── HUBERT.py                   # Legacy training script
│   ├── enhanced_ser_model.py       # Model architecture (CmGI)
│   ├── test_only.py                # Inference script
│   ├── process.py                  # Data preprocessing
│   └── split_iemocap.py            # Dataset splitting
│
├── data/                           # Dataset directory
│   ├── README.md                   # Dataset documentation
│   ├── IEMOCAP/                    # Raw IEMOCAP data (not tracked)
│   ├── iemocap_all.json            # Processed dataset (not tracked)
│   ├── iemocap_train.json          # Training set (not tracked)
│   ├── iemocap_val.json            # Validation set (not tracked)
│   └── iemocap_test.json           # Test set (not tracked)
│
├── models/                         # Model information
│   └── README.md                   # Pre-trained model documentation
│
├── checkpoints/                    # Model checkpoints
│   └── .gitkeep                    # Keep directory in git
│   └── *.pt                        # PyTorch checkpoints (not tracked)
│
├── scripts/                        # Utility scripts
│   └── submit_hubert.sh            # SLURM job submission script
│
└── docs/                           # Documentation
    └── paper.pdf                   # Research paper
```

## File Descriptions

### Root Directory

#### Configuration Files

- **`.gitignore`**: Specifies files and directories to ignore in version control
  - Python cache files
  - Model checkpoints
  - Dataset files
  - Log files
  - Virtual environments

- **`requirements.txt`**: Python package dependencies
  - PyTorch and related libraries
  - Transformers library
  - Audio processing libraries
  - Utilities and visualization tools

- **`LICENSE`**: MIT License for the project

#### Documentation

- **`README.md`**: Main project documentation
  - Abstract and overview
  - Installation instructions
  - Usage examples
  - Results and performance
  - Citation information

- **`SETUP.md`**: Detailed setup guide
  - System requirements
  - Step-by-step installation
  - Dataset preparation
  - Configuration options
  - Troubleshooting

- **`CONTRIBUTING.md`**: Guidelines for contributors
  - How to contribute
  - Code style guidelines
  - Testing requirements
  - Pull request process

### Source Code (`src/`)

#### Training Scripts

- **`train.py`**: Main training script (recommended)
  - Clean, well-documented implementation
  - Full training pipeline
  - Validation and testing
  - Checkpoint saving
  - Logging functionality

- **`HUBERT.py`**: Legacy training script
  - Original implementation
  - Contains duplicate imports
  - Kept for reference

#### Model Architecture

- **`enhanced_ser_model.py`**: Model definitions
  - `TemporalGatedFusion`: Gating mechanism
  - `CmGI`: Cross-modal gated interaction
  - `EnhancedSERModel`: Complete SER model
  - Implements CmGI baseline (not simplified version)

#### Data Processing

- **`process.py`**: Data preprocessing script
  - Reads raw IEMOCAP data
  - Extracts audio files and labels
  - Creates `iemocap_all.json`

- **`split_iemocap.py`**: Dataset splitting
  - Splits data by session
  - Creates train/val/test sets
  - Follows paper's methodology

#### Inference

- **`test_only.py`**: Testing script
  - Loads trained model
  - Evaluates on test set
  - Reports accuracy metrics

### Data (`data/`)

- **`README.md`**: Dataset documentation
  - How to obtain IEMOCAP
  - Dataset structure
  - Processing instructions
  - Statistics and format

- **`IEMOCAP/`**: Raw dataset (not tracked)
  - Requires permission from USC
  - ~12 hours of multimodal data
  - 5 sessions, 10 speakers

- **`*.json`**: Processed datasets (not tracked)
  - Contain audio paths and labels
  - Created by preprocessing scripts

### Models (`models/`)

- **`README.md`**: Pre-trained model information
  - HuBERT, BERT, Wav2vec 2.0
  - Download instructions
  - Model specifications
  - GPU requirements

### Checkpoints (`checkpoints/`)

- **`.gitkeep`**: Keeps directory in version control
- **`*.pt`**: Model checkpoints (not tracked)
  - Saved during training
  - Named with epoch and accuracy
  - Can be loaded for inference

### Scripts (`scripts/`)

- **`submit_hubert.sh`**: SLURM job script
  - GPU allocation
  - Environment setup
  - Job submission parameters
  - For cluster training

### Documentation (`docs/`)

- **`paper.pdf`**: Research paper
  - Full methodology
  - Experimental results
  - References and citations

## File Relationships

### Training Pipeline Flow

```
1. Data Preparation:
   process.py → iemocap_all.json
   split_iemocap.py → train/val/test.json

2. Training:
   train.py loads:
   - enhanced_ser_model.py (model)
   - iemocap_*.json (data)
   - HuBERT, BERT, Wav2vec2 (pretrained)
   
   Outputs:
   - checkpoints/*.pt
   - training_log_*.txt

3. Testing:
   test_only.py loads:
   - enhanced_ser_model.py
   - checkpoints/*.pt
   - iemocap_test.json
   
   Outputs:
   - Test accuracy metrics
```

### Model Architecture Flow

```
Audio Input → HuBERT → Acoustic Features (1024-dim)
                                ↓
                           CmGI Fusion ← Semantic Features (1024-dim)
                                ↓            ↑
Text Input → Wav2vec2 → ASR → BERT → Projection (768→1024)
                                ↓
                         Average Pooling
                                ↓
                         Concatenation (2048-dim)
                                ↓
                      Fully Connected Layer
                                ↓
                        Emotion Logits (4 classes)
```

## Key Design Decisions

### Why This Structure?

1. **Separation of Concerns**
   - Source code in `src/`
   - Data in `data/`
   - Documentation in `docs/`

2. **Reproducibility**
   - Clear preprocessing scripts
   - Saved checkpoints
   - Logged training process

3. **Ease of Use**
   - Comprehensive README
   - Setup guide
   - Example scripts

4. **Maintainability**
   - Modular code structure
   - Clear file naming
   - Documentation at all levels

### Not Tracked in Git

- Dataset files (too large, requires permission)
- Model checkpoints (too large, user-generated)
- Log files (user-generated)
- Cache files (auto-generated)
- Virtual environments (user-specific)

### Tracked in Git

- Source code
- Documentation
- Configuration files
- Scripts
- Paper (PDF)

## Future Extensions

Potential additions to the structure:

```
speech-emotion-recognition/
├── experiments/              # Experiment tracking
│   ├── config_1.yaml
│   └── results_1.json
│
├── notebooks/                # Jupyter notebooks
│   ├── analysis.ipynb
│   └── visualization.ipynb
│
├── tests/                    # Unit tests
│   ├── test_model.py
│   ├── test_data.py
│   └── test_utils.py
│
└── utils/                    # Helper functions
    ├── metrics.py
    ├── visualization.py
    └── augmentation.py
```

## Maintenance Notes

### Adding New Files

When adding new files:
1. Place in appropriate directory
2. Update relevant README
3. Add to .gitignore if needed
4. Document in this file

### Modifying Structure

If changing the structure:
1. Update this document
2. Update all path references in code
3. Test all scripts
4. Update setup instructions

## Version Control

Tracked files:
- Use semantic versioning for releases
- Keep commit history clean
- Document major changes

Ignored files:
- Listed in `.gitignore`
- Document important ones here
- Provide instructions for obtaining them
