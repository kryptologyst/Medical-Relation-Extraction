#!/usr/bin/env python3
"""Example usage of Medical Relation Extraction models."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models import RuleBasedRelationExtractor, TransformerRelationExtractor, RelationExtractionConfig
from data import SyntheticClinicalDataset, DataProcessor
from metrics import RelationExtractionMetrics


def example_rule_based():
    """Example of using rule-based relation extraction."""
    print("=== Rule-Based Relation Extraction Example ===")
    
    try:
        # Initialize extractor
        extractor = RuleBasedRelationExtractor()
        
        # Sample clinical text
        text = "The patient was prescribed metformin to manage type 2 diabetes and later developed nausea."
        
        print(f"Input text: {text}")
        print()
        
        # Extract entities
        entities = extractor.extract_entities(text)
        print("Extracted entities:")
        for entity in entities:
            print(f"  - {entity[0]} ({entity[1]})")
        print()
        
        # Extract relations
        relations = extractor.extract_relations(text)
        print("Extracted relations:")
        for relation in relations:
            print(f"  - {relation['entity1']} → {relation['entity2']} ({relation['relation']}, confidence: {relation['confidence']:.3f})")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install the required spaCy model:")
        print("python -m spacy download en_ner_bc5cdr_md")


def example_synthetic_data():
    """Example of generating synthetic clinical data."""
    print("=== Synthetic Clinical Data Example ===")
    
    # Generate synthetic dataset
    dataset_generator = SyntheticClinicalDataset()
    samples = dataset_generator.generate_dataset(3)
    
    print("Generated synthetic clinical samples:")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Text: {sample.text}")
        print(f"  Entities: {len(sample.entities)}")
        for entity in sample.entities:
            print(f"    - {entity['text']} ({entity['label']})")
        print(f"  Relations: {len(sample.relations)}")
        for relation in sample.relations:
            print(f"    - {relation['entity1']} → {relation['entity2']} ({relation['relation']})")
        print(f"  De-identified: {sample.deidentified_text}")
    print()


def example_data_processing():
    """Example of data processing and de-identification."""
    print("=== Data Processing Example ===")
    
    # Sample text with PHI
    text = "Patient John Doe (DOB: 01/15/1985, MRN: 123456) was prescribed metformin. Contact: john.doe@email.com"
    
    print(f"Original text: {text}")
    
    # Process with de-identification
    processor = DataProcessor(deidentify=True)
    processed_text = processor.process_text(text)
    
    print(f"De-identified text: {processed_text}")
    print()


def example_evaluation():
    """Example of evaluation metrics."""
    print("=== Evaluation Metrics Example ===")
    
    # Create synthetic data
    dataset_generator = SyntheticClinicalDataset()
    samples = dataset_generator.generate_dataset(5)
    
    # Initialize metrics
    relation_types = ["treats", "causes", "prevents", "worsens", "side_effect_of"]
    metrics = RelationExtractionMetrics(relation_types)
    
    # Simulate predictions (perfect predictions for demo)
    for sample in samples:
        for relation in sample.relations:
            metrics.add_prediction(
                relation['relation'],  # true relation
                relation['relation'],  # predicted relation (perfect)
                relation['confidence'],  # confidence
                relation['entity1'],
                relation['entity2']
            )
    
    # Calculate metrics
    results = metrics.calculate_metrics()
    
    print("Evaluation Results:")
    print(f"  Accuracy: {results['overall']['accuracy']:.3f}")
    print(f"  Precision: {results['overall']['precision']:.3f}")
    print(f"  Recall: {results['overall']['recall']:.3f}")
    print(f"  F1-Score: {results['overall']['f1']:.3f}")
    print()


def main():
    """Main function to run all examples."""
    print("Medical Relation Extraction - Usage Examples")
    print("=" * 50)
    print()
    
    # Run examples
    example_synthetic_data()
    example_data_processing()
    example_evaluation()
    example_rule_based()
    
    print("=" * 50)
    print("Examples completed!")
    print()
    print("To run the interactive demo:")
    print("  python run_demo.py")
    print()
    print("To train a model:")
    print("  python src/train.py")
    print()
    print("To evaluate models:")
    print("  python src/evaluate.py")


if __name__ == "__main__":
    main()
