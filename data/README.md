# Data Directory

This directory contains the IEMOCAP dataset and processed data files.

## IEMOCAP Dataset

### Obtaining the Dataset

1. Visit the IEMOCAP project page: https://sail.usc.edu/iemocap/
2. Request access by filling out the form
3. Wait for approval from USC (typically takes a few days to weeks)
4. Download the dataset after receiving approval

### Dataset Structure

After downloading, the IEMOCAP dataset should be organized as follows:

```
data/
├── IEMOCAP/
│   ├── Session1/
│   │   ├── dialog/
│   │   │   ├── wav/
│   │   │   └── transcriptions/
│   │   └── sentences/
│   │       └── wav/
│   ├── Session2/
│   ├── Session3/
│   ├── Session4/
│   └── Session5/
└── processed/
    ├── iemocap_all.json
    ├── iemocap_train.json
    ├── iemocap_val.json
    └── iemocap_test.json
```

## Processing the Dataset

### Step 1: Extract Audio and Labels

Run the preprocessing script to extract audio files and emotion labels:

```bash
cd ../src
python process.py
```

This creates `iemocap_all.json` containing:
- Audio file paths
- Emotion labels
- Session information
- Transcriptions (if available)

### Step 2: Split into Train/Val/Test

Split the dataset according to the paper's methodology:

```bash
python split_iemocap.py
```

This creates:
- `iemocap_train.json`: Sessions 1, 3, 4, 5 (training)
- `iemocap_val.json`: 20% of training sessions (validation)
- `iemocap_test.json`: Session 2 (testing)

## Dataset Statistics

### Original IEMOCAP
- Total duration: ~12 hours
- Total utterances: 10,039
- Speakers: 10 (5 male, 5 female)
- Sessions: 5
- Original emotion categories: 9 (neutral, happiness, sadness, anger, surprise, fear, disgust, frustration, excited, other)

### Processed Dataset (4-class)
- Emotion classes: 4 (happy, sad, angry, neutral)
- Mapping: "excited" merged with "happy"
- Total samples: ~5,531 utterances
- Training: ~2,230 samples
- Validation: ~558 samples
- Test: ~678 samples

## JSON Format

Each entry in the JSON files has the following structure:

```json
{
  "audio_path": "path/to/audio.wav",
  "emotion": "happy",
  "session": "Session1",
  "speaker": "Ses01F_impro01",
  "transcription": "text of the utterance"
}
```

## Notes

- Audio files are in WAV format, 16kHz sampling rate
- All audio is resampled to 16kHz during data loading
- Stereo audio is converted to mono by averaging channels
- Ground-truth transcriptions are available but the model uses ASR predictions

## Citation

If you use the IEMOCAP dataset, please cite:

```bibtex
@article{busso2008iemocap,
  title={IEMOCAP: Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal={Language resources and evaluation},
  volume={42},
  number={4},
  pages={335--359},
  year={2008},
  publisher={Springer}
}
```
