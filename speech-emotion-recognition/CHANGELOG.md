# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### To Be Implemented
- Simplified single-layer fusion architecture as proposed in paper
- Data augmentation (noise injection, time stretching, pitch shifting)
- Regularization techniques (dropout, weight decay, early stopping)
- K-fold cross-validation support
- Confusion matrix visualization
- Training curve plotting
- TensorBoard integration
- Mixed precision training
- Model export (ONNX, TorchScript)

## [1.0.0] - 2024-12-01

### Added
- Initial repository setup with complete project structure
- CmGI baseline model implementation (Gao et al.)
- Training pipeline with HuBERT, BERT, and Wav2vec 2.0
- IEMOCAP dataset preprocessing scripts
- Comprehensive documentation:
  - README.md with project overview
  - SETUP.md with installation guide
  - CONTRIBUTING.md with contribution guidelines
  - PROJECT_STRUCTURE.md detailing repository organization
  - Model and data documentation
- SLURM job submission script for cluster training
- Requirements file with all dependencies
- MIT License
- .gitignore for Python projects

### Model Architecture
- HuBERT Large for acoustic feature extraction (1024-dim)
- BERT Base for semantic feature extraction (768→1024-dim)
- Wav2vec 2.0 for automatic speech recognition
- Cross-modal Gated Interaction (CmGI) fusion
- Temporal-aware gating mechanism
- 4-class emotion classification (happy, sad, angry, neutral)

### Performance
- Training accuracy: 95.43% (epoch 6)
- Validation accuracy: 73-75% (plateaus after epoch 3)
- Known issue: Overfitting after epoch 3

### Dataset
- IEMOCAP dataset support
- Session-based train/val/test split
- Session 2 held out for testing
- Automatic ASR transcription generation
- 5,531 total samples (2,230 train, 558 val, 678 test)

### Documentation
- Research paper included (docs/paper.pdf)
- Detailed API documentation in code
- Troubleshooting guide
- Performance optimization tips

## [0.1.0] - 2024-03-27

### Development Version
- Initial model training experiments
- Basic training script (HUBERT.py)
- Preliminary results on IEMOCAP
- Training logs showing overfitting pattern

---

## Version History Notes

### Version 1.0.0 Notes

This is the first public release, corresponding to the research paper submission.
The implementation includes the CmGI baseline model but not yet the simplified
single-layer architecture proposed in the paper.

Key metrics achieved:
- Comparable to Yu et al. (69.65% → 74.9%)
- Lower than CmGI baseline (79.5%)
- Lower than MMER multi-task (81.2%)

Known limitations:
- Overfitting after epoch 3-6
- No regularization implemented
- No data augmentation
- Small dataset (5,531 samples)

### Roadmap for 2.0.0

Priority features for next major release:
1. Implement proposed simplified architecture
2. Add dropout and weight decay
3. Implement data augmentation
4. Add early stopping
5. Cross-validation support
6. Performance improvements

### Breaking Changes

None expected between 1.x versions. Major architectural changes will be 
introduced in 2.0.0.

### Compatibility

- Python: 3.8+
- PyTorch: 1.10+
- Transformers: 4.20+
- CUDA: 11.0+ (optional, for GPU)

### Contributors

- Xuefei Bian
- Hao-wei Liang
- Tiantian Zhang

### Citations

If you use this code, please cite:

```bibtex
@article{bian2024single,
  title={Single-Layered Fusion in Uni-Task Multimodal Speech Emotion Recognition Models},
  author={Bian, Xuefei and Liang, Hao-wei and Zhang, Tiantian},
  year={2024}
}
```
