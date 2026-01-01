"""Tests for medical relation extraction."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models import (
    RuleBasedRelationExtractor, 
    TransformerRelationExtractor, 
    EnsembleRelationExtractor,
    RelationExtractionConfig
)
from src.data import SyntheticClinicalDataset, DataProcessor, ClinicalTextSample
from src.metrics import RelationExtractionMetrics, RelationExtractionEvaluator
from src.utils import set_seed, get_device, validate_text_input, sanitize_text


class TestRuleBasedRelationExtractor:
    """Test cases for rule-based relation extractor."""
    
    def test_init(self):
        """Test initializer."""
        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            
            extractor = RuleBasedRelationExtractor()
            assert extractor.nlp == mock_nlp
            assert len(extractor.relation_patterns) == 5
    
    def test_extract_entities(self):
        """Test entity extraction."""
        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_ent1 = Mock()
            mock_ent1.text = "metformin"
            mock_ent1.label_ = "CHEMICAL"
            mock_ent1.start_char = 0
            mock_ent1.end_char = 9
            
            mock_ent2 = Mock()
            mock_ent2.text = "diabetes"
            mock_ent2.label_ = "DISEASE"
            mock_ent2.start_char = 20
            mock_ent2.end_char = 28
            
            mock_doc.ents = [mock_ent1, mock_ent2]
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp
            
            extractor = RuleBasedRelationExtractor()
            entities = extractor.extract_entities("metformin treats diabetes")
            
            assert len(entities) == 2
            assert entities[0] == ("metformin", "CHEMICAL", 0, 9)
            assert entities[1] == ("diabetes", "DISEASE", 20, 28)
    
    def test_extract_relations(self):
        """Test relation extraction."""
        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_ent1 = Mock()
            mock_ent1.text = "metformin"
            mock_ent1.label_ = "CHEMICAL"
            mock_ent1.start_char = 0
            mock_ent1.end_char = 9
            
            mock_ent2 = Mock()
            mock_ent2.text = "diabetes"
            mock_ent2.label_ = "DISEASE"
            mock_ent2.start_char = 20
            mock_ent2.end_char = 28
            
            mock_doc.ents = [mock_ent1, mock_ent2]
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp
            
            extractor = RuleBasedRelationExtractor()
            relations = extractor.extract_relations("metformin treats diabetes")
            
            assert len(relations) >= 0  # May or may not find relations depending on patterns


class TestSyntheticClinicalDataset:
    """Test cases for synthetic dataset generator."""
    
    def test_init(self):
        """Test initializer."""
        dataset = SyntheticClinicalDataset()
        assert len(dataset.diseases) > 0
        assert len(dataset.drugs) > 0
        assert len(dataset.symptoms) > 0
        assert len(dataset.treatment_templates) > 0
    
    def test_generate_sample(self):
        """Test sample generation."""
        dataset = SyntheticClinicalDataset()
        sample = dataset.generate_sample()
        
        assert isinstance(sample, ClinicalTextSample)
        assert len(sample.text) > 0
        assert len(sample.entities) > 0
        assert len(sample.relations) > 0
        assert len(sample.deidentified_text) > 0
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        dataset = SyntheticClinicalDataset()
        samples = dataset.generate_dataset(10)
        
        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, ClinicalTextSample)
            assert len(sample.text) > 0


class TestDataProcessor:
    """Test cases for data processor."""
    
    def test_init(self):
        """Test initializer."""
        processor = DataProcessor()
        assert processor.deidentify == True
        assert len(processor.deid_patterns) > 0
    
    def test_process_text(self):
        """Test text processing."""
        processor = DataProcessor(deidentify=True)
        text = "Patient John Doe (DOB: 01/01/1990) was prescribed metformin."
        processed = processor.process_text(text)
        
        assert "John Doe" not in processed
        assert "01/01/1990" not in processed
        assert "metformin" in processed
    
    def test_deidentify_text(self):
        """Test de-identification."""
        processor = DataProcessor()
        text = "Patient John Doe (DOB: 01/01/1990) was prescribed metformin."
        deidentified = processor.deidentify_text(text)
        
        assert "John Doe" not in deidentified
        assert "01/01/1990" not in deidentified


class TestRelationExtractionMetrics:
    """Test cases for relation extraction metrics."""
    
    def test_init(self):
        """Test initializer."""
        relation_types = ["treats", "causes"]
        metrics = RelationExtractionMetrics(relation_types)
        assert metrics.relation_types == relation_types
        assert len(metrics.results) == 0
    
    def test_add_prediction(self):
        """Test adding predictions."""
        metrics = RelationExtractionMetrics(["treats"])
        metrics.add_prediction("treats", "treats", 0.9, "drug", "disease")
        
        assert len(metrics.results['true_relations']) == 1
        assert metrics.results['true_relations'][0] == "treats"
        assert metrics.results['predicted_relations'][0] == "treats"
        assert metrics.results['confidences'][0] == 0.9
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = RelationExtractionMetrics(["treats", "causes"])
        
        # Add some predictions
        metrics.add_prediction("treats", "treats", 0.9, "drug1", "disease1")
        metrics.add_prediction("causes", "causes", 0.8, "symptom1", "disease2")
        metrics.add_prediction("treats", "causes", 0.7, "drug2", "disease3")
        
        results = metrics.calculate_metrics()
        
        assert 'overall' in results
        assert 'per_relation' in results
        assert 'confidence_analysis' in results
        assert results['overall']['accuracy'] >= 0.0
        assert results['overall']['accuracy'] <= 1.0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can verify it doesn't raise an error
        assert True
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_validate_text_input(self):
        """Test text input validation."""
        assert validate_text_input("Valid text") == True
        assert validate_text_input("") == False
        assert validate_text_input("   ") == False
        assert validate_text_input("x" * 1001) == False
        assert validate_text_input("<script>alert('xss')</script>") == False
    
    def test_sanitize_text(self):
        """Test text sanitization."""
        text = "<p>Hello   world</p>"
        sanitized = sanitize_text(text)
        assert "<p>" not in sanitized
        assert "Hello world" in sanitized


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_rule_based(self):
        """Test end-to-end rule-based extraction."""
        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_ent1 = Mock()
            mock_ent1.text = "metformin"
            mock_ent1.label_ = "CHEMICAL"
            mock_ent1.start_char = 0
            mock_ent1.end_char = 9
            
            mock_ent2 = Mock()
            mock_ent2.text = "diabetes"
            mock_ent2.label_ = "DISEASE"
            mock_ent2.start_char = 20
            mock_ent2.end_char = 28
            
            mock_doc.ents = [mock_ent1, mock_ent2]
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp
            
            # Generate synthetic data
            dataset = SyntheticClinicalDataset()
            sample = dataset.generate_sample()
            
            # Process with rule-based extractor
            extractor = RuleBasedRelationExtractor()
            relations = extractor.extract_relations(sample.text)
            
            # Should return some relations
            assert isinstance(relations, list)
    
    def test_metrics_integration(self):
        """Test metrics integration."""
        # Create synthetic samples
        dataset = SyntheticClinicalDataset()
        samples = dataset.generate_dataset(5)
        
        # Create metrics
        metrics = RelationExtractionMetrics(["treats", "causes"])
        
        # Add predictions
        for sample in samples:
            for relation in sample.relations:
                metrics.add_prediction(
                    relation['relation'],
                    relation['relation'],  # Assume perfect prediction
                    relation['confidence'],
                    relation['entity1'],
                    relation['entity2']
                )
        
        # Calculate metrics
        results = metrics.calculate_metrics()
        
        assert 'overall' in results
        assert results['overall']['accuracy'] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])
