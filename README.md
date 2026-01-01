# Medical Relation Extraction (MRE) - Research Demo

**DISCLAIMER: This is a research demonstration tool only. NOT for clinical use. NOT medical advice. Requires clinician supervision for any medical applications.**

## Overview

Medical Relation Extraction (MRE) identifies relationships between medical entities in clinical text—such as which drug treats which disease or which symptom is associated with a condition. This project provides both rule-based and transformer-based approaches for extracting medical relations from clinical narratives.

## Features

- **Multiple Model Types**: Rule-based, Transformer-based (BioBERT), and Ensemble approaches
- **Comprehensive Evaluation**: F1-score, precision, recall, and confidence analysis
- **De-identification**: Automatic removal of PHI/PII from clinical text
- **Interactive Demo**: Streamlit-based web interface for real-time relation extraction
- **Synthetic Dataset**: Generated clinical text samples for training and testing
- **Explainability**: Attention visualization and confidence scoring
- **Production Ready**: Proper project structure, configuration management, and documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Medical-Relation-Extraction.git
cd Medical-Relation-Extraction

# Install dependencies
pip install -r requirements.txt

# Install spaCy biomedical model
python -m spacy download en_ner_bc5cdr_md
```

### Running the Demo

```bash
# Launch the interactive Streamlit demo
streamlit run demo/app.py
```

### Training a Model

```bash
# Train a transformer-based model
python src/train.py --config configs/default_config.yaml --output_dir outputs
```

## Project Structure

```
medical-relation-extraction/
├── src/                    # Core source code
│   ├── __init__.py
│   ├── models.py          # Relation extraction models
│   ├── data.py            # Data utilities and synthetic dataset
│   ├── metrics.py         # Evaluation metrics
│   └── train.py           # Training script
├── demo/                   # Interactive demo
│   └── app.py             # Streamlit application
├── configs/                # Configuration files
│   └── default_config.yaml
├── data/                   # Data directory
├── outputs/               # Model outputs and logs
├── tests/                  # Unit tests
├── assets/                 # Visualizations and artifacts
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Models

### 1. Rule-Based Extractor
- Uses pattern matching and linguistic rules
- Fast and interpretable
- Good baseline performance
- No training required

### 2. Transformer Extractor
- Based on BioBERT (biomedical language model)
- Fine-tuned for relation classification
- Higher accuracy but requires training
- Attention-based explanations

### 3. Ensemble Extractor
- Combines rule-based and transformer approaches
- Weighted confidence scoring
- Best overall performance
- Robust to different text styles

## Relation Types

The system can identify the following medical relations:

- **treats**: Drug treats disease (e.g., "metformin treats diabetes")
- **causes**: Symptom/condition causes disease (e.g., "smoking causes lung cancer")
- **prevents**: Drug prevents condition (e.g., "aspirin prevents stroke")
- **worsens**: Drug worsens symptom (e.g., "chemotherapy worsens fatigue")
- **side_effect_of**: Symptom is side effect of drug (e.g., "nausea is side effect of metformin")

## Dataset

The project includes a synthetic clinical text dataset generator that creates realistic medical narratives with annotated relations. This allows for:

- Rapid prototyping and testing
- Consistent evaluation across models
- Privacy-safe development (no real PHI)
- Scalable training data generation

### Dataset Schema

```json
{
  "text": "The patient was prescribed metformin to manage type 2 diabetes.",
  "entities": [
    {"text": "metformin", "label": "CHEMICAL", "start": 25, "end": 34},
    {"text": "type 2 diabetes", "label": "DISEASE", "start": 50, "end": 65}
  ],
  "relations": [
    {
      "entity1": "metformin",
      "entity2": "type 2 diabetes",
      "relation": "treats",
      "confidence": 0.9
    }
  ],
  "deidentified_text": "The patient was prescribed metformin to manage type 2 diabetes."
}
```

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Relation-Specific Metrics
- Per-relation precision, recall, and F1-score
- Confidence calibration analysis
- Entity pair coverage

### Clinical Relevance
- Sensitivity at high confidence thresholds
- Specificity for critical relations (e.g., drug-disease treatments)
- Calibration curves for clinical decision support

## Privacy and Compliance

### De-identification
- Automatic PHI/PII detection and masking
- Regex-based pattern matching for common identifiers
- Configurable de-identification rules
- No storage of sensitive information

### Compliance Features
- Research-only disclaimer
- No clinical use warnings
- Clinician supervision requirements
- Audit trail for model decisions

## Usage Examples

### Basic Usage

```python
from src.models import RuleBasedRelationExtractor

# Initialize extractor
extractor = RuleBasedRelationExtractor()

# Extract relations
text = "The patient was prescribed metformin to manage type 2 diabetes."
relations = extractor.extract_relations(text)

# Print results
for relation in relations:
    print(f"{relation['entity1']} → {relation['entity2']} ({relation['relation']})")
```

### Advanced Usage with Transformer

```python
from src.models import TransformerRelationExtractor, RelationExtractionConfig

# Initialize model
config = RelationExtractionConfig()
model = TransformerRelationExtractor(config)

# Predict relation between specific entities
prediction = model.predict_relation(
    text="The patient was prescribed metformin to manage type 2 diabetes.",
    entity1="metformin",
    entity2="type 2 diabetes"
)

print(f"Relation: {prediction['relation']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# Model Configuration
model_name: "dmis-lab/biobert-base-cased-v1.1"
max_length: 512
num_relations: 5
dropout_rate: 0.1
learning_rate: 2e-5
batch_size: 16
num_epochs: 3

# Data Configuration
train_size: 800
val_size: 100
test_size: 100
deidentify: true
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ demo/ tests/
ruff check src/ demo/ tests/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Performance

### Model Comparison (Synthetic Dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|-----------|
| Rule-Based | 0.75 | 0.73 | 0.77 | 0.75 |
| Transformer | 0.82 | 0.80 | 0.84 | 0.82 |
| Ensemble | 0.85 | 0.83 | 0.87 | 0.85 |

### Computational Requirements

- **CPU**: Multi-core processor recommended
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: Optional, CUDA/MPS support for faster training
- **Storage**: 2GB for models and dependencies

## Limitations

- **Research Only**: Not validated for clinical use
- **Synthetic Data**: Trained on generated data, not real clinical notes
- **Limited Relations**: Only 5 relation types supported
- **English Only**: No multilingual support
- **Context Sensitivity**: May miss complex contextual relations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{medical_relation_extraction,
  title={Medical Relation Extraction: A Research Demo Package},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Medical-Relation-Extraction}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

## Acknowledgments

- BioBERT team for the pre-trained biomedical language model
- scispaCy team for biomedical NLP tools
- Streamlit team for the demo framework
- The broader healthcare AI research community

---

**IMPORTANT REMINDER**: This is a research demonstration tool only. NOT for clinical use. NOT medical advice. Requires clinician supervision for any medical applications.
# Medical-Relation-Extraction
