# Contributing to Speech Emotion Recognition

Thank you for considering contributing to this project! This document outlines the guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU, etc.)
- Error messages or logs

### Suggesting Enhancements

We welcome suggestions for improvements:
- Open an issue with the enhancement tag
- Describe the feature and its benefits
- Provide examples of how it would be used

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install development dependencies:
```bash
pip install pytest black flake8 mypy
```

## Code Style

### Python

- Follow PEP 8 guidelines
- Use Black for code formatting: `black src/`
- Use flake8 for linting: `flake8 src/`
- Use type hints where possible
- Maximum line length: 100 characters

### Documentation

- Use Google-style docstrings
- Document all public functions and classes
- Include type hints in function signatures
- Provide examples in docstrings for complex functions

Example:
```python
def process_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Process audio file and return tensor representation.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sampling rate in Hz
        
    Returns:
        Processed audio tensor of shape (samples,)
        
    Example:
        >>> audio = process_audio("data/sample.wav")
        >>> audio.shape
        torch.Size([160000])
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
pytest tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures for common setup
- Aim for >80% code coverage

Example:
```python
def test_model_forward_pass():
    """Test that model forward pass produces correct output shape."""
    model = EnhancedSERModel(hubert_model, bert_model, num_classes=4)
    audio_input = torch.randn(2, 16000)
    text_input = {"input_ids": torch.randint(0, 1000, (2, 50)),
                  "attention_mask": torch.ones(2, 50)}
    
    output = model(audio_input, text_input)
    
    assert output.shape == (2, 4)
```

## Commit Messages

Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example:
```
feat: add data augmentation for audio samples

- Implement time stretching
- Add noise injection
- Include pitch shifting
```

## Project Structure Guidelines

When adding new files:

```
src/               # Source code
├── models/        # Model architectures
├── data/          # Data loading and processing
├── training/      # Training scripts and utilities
└── utils/         # Helper functions

tests/             # Unit tests
├── test_models/
├── test_data/
└── test_utils/

docs/              # Documentation
└── api/           # API documentation
```

## Areas for Contribution

Current priorities:

1. **Regularization**: Implement dropout, weight decay, early stopping
2. **Data Augmentation**: Add audio augmentation techniques
3. **Simplified Model**: Implement the proposed single-layer fusion
4. **Cross-validation**: Add k-fold cross-validation support
5. **Visualization**: Add training curves and confusion matrices
6. **Documentation**: Improve code documentation and tutorials
7. **Testing**: Increase test coverage

## Code Review Process

1. All submissions require review
2. Reviewers will check:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Performance implications
3. Address review comments
4. Maintain backwards compatibility when possible

## Questions?

Feel free to open an issue for:
- Clarification on contribution guidelines
- Technical questions about the codebase
- Discussion on potential features

Thank you for contributing!
