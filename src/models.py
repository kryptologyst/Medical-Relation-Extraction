"""Core models for medical relation extraction."""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    BertForSequenceClassification
)
import spacy
from dataclasses import dataclass
import numpy as np


@dataclass
class RelationExtractionConfig:
    """Configuration for relation extraction models."""
    model_name: str = "dmis-lab/biobert-base-cased-v1.1"
    max_length: int = 512
    num_relations: int = 5  # treats, causes, prevents, worsens, side_effect_of
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3


class RuleBasedRelationExtractor:
    """Rule-based relation extraction using spaCy and pattern matching."""
    
    def __init__(self, model_name: str = "en_ner_bc5cdr_md"):
        """Initialize the rule-based extractor.
        
        Args:
            model_name: Name of the spaCy biomedical NER model to use.
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Please install it first:")
            print(f"python -m spacy download {model_name}")
            raise
        
        # Define relation patterns
        self.relation_patterns = {
            "treats": ["prescribed", "treat", "manage", "therapy", "medication"],
            "causes": ["caused by", "due to", "resulting from", "triggered by"],
            "prevents": ["prevent", "avoid", "protect against", "prophylaxis"],
            "worsens": ["worsen", "exacerbate", "aggravate", "deteriorate"],
            "side_effect_of": ["developed", "experienced", "adverse effect", "complication"]
        }
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract medical entities from text.
        
        Args:
            text: Input clinical text.
            
        Returns:
            List of (entity_text, label, start_char, end_char) tuples.
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
        return entities
    
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relations between entities using rule-based patterns.
        
        Args:
            text: Input clinical text.
            
        Returns:
            List of relation dictionaries with entity pairs and relation types.
        """
        entities = self.extract_entities(text)
        relations = []
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                ent1, label1, start1, end1 = entities[i]
                ent2, label2, start2, end2 = entities[j]
                
                # Determine span between entities
                span_start = min(start1, start2)
                span_end = max(end1, end2)
                span_text = text[span_start:span_end].lower()
                
                # Check for relation patterns
                for relation_type, patterns in self.relation_patterns.items():
                    if any(pattern in span_text for pattern in patterns):
                        relation = {
                            "entity1": ent1,
                            "entity2": ent2,
                            "label1": label1,
                            "label2": label2,
                            "relation": relation_type,
                            "confidence": 0.8,  # Rule-based confidence
                            "span": span_text
                        }
                        relations.append(relation)
        
        return relations


class TransformerRelationExtractor(nn.Module):
    """Transformer-based relation extraction model."""
    
    def __init__(self, config: RelationExtractionConfig):
        """Initialize the transformer model.
        
        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.config = config
        
        # Load pre-trained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Classification head
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size // 2, config.num_relations)
        )
        
        # Relation type mapping
        self.relation_types = ["treats", "causes", "prevents", "worsens", "side_effect_of"]
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequences.
            attention_mask: Attention mask for input sequences.
            
        Returns:
            Logits for relation classification.
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
    
    def predict_relation(self, text: str, entity1: str, entity2: str) -> Dict[str, Any]:
        """Predict relation between two entities.
        
        Args:
            text: Input clinical text.
            entity1: First entity.
            entity2: Second entity.
            
        Returns:
            Dictionary with predicted relation and confidence.
        """
        # Create input with entity markers
        marked_text = f"[E1]{entity1}[/E1] {text} [E2]{entity2}[/E2]"
        
        # Tokenize
        inputs = self.tokenizer(
            marked_text,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Predict
        self.eval()
        with torch.no_grad():
            logits = self.forward(inputs["input_ids"], inputs["attention_mask"])
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "relation": self.relation_types[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                rel_type: prob.item() 
                for rel_type, prob in zip(self.relation_types, probabilities[0])
            }
        }


class EnsembleRelationExtractor:
    """Ensemble of rule-based and transformer-based extractors."""
    
    def __init__(self, config: RelationExtractionConfig):
        """Initialize the ensemble extractor.
        
        Args:
            config: Configuration object.
        """
        self.rule_extractor = RuleBasedRelationExtractor()
        self.transformer_extractor = TransformerRelationExtractor(config)
        self.config = config
    
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relations using ensemble approach.
        
        Args:
            text: Input clinical text.
            
        Returns:
            List of extracted relations with ensemble confidence.
        """
        # Get rule-based relations
        rule_relations = self.rule_extractor.extract_relations(text)
        
        # Get entities for transformer predictions
        entities = self.rule_extractor.extract_entities(text)
        
        ensemble_relations = []
        
        # For each entity pair, get transformer prediction
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                ent1, label1, _, _ = entities[i]
                ent2, label2, _, _ = entities[j]
                
                # Get transformer prediction
                transformer_pred = self.transformer_extractor.predict_relation(
                    text, ent1, ent2
                )
                
                # Combine with rule-based confidence
                rule_confidence = 0.0
                for rule_rel in rule_relations:
                    if (rule_rel["entity1"] == ent1 and rule_rel["entity2"] == ent2) or \
                       (rule_rel["entity1"] == ent2 and rule_rel["entity2"] == ent1):
                        rule_confidence = rule_rel["confidence"]
                        break
                
                # Ensemble confidence (weighted average)
                ensemble_confidence = 0.6 * transformer_pred["confidence"] + 0.4 * rule_confidence
                
                if ensemble_confidence > 0.5:  # Threshold for inclusion
                    relation = {
                        "entity1": ent1,
                        "entity2": ent2,
                        "label1": label1,
                        "label2": label2,
                        "relation": transformer_pred["relation"],
                        "confidence": ensemble_confidence,
                        "rule_confidence": rule_confidence,
                        "transformer_confidence": transformer_pred["confidence"],
                        "all_probabilities": transformer_pred["all_probabilities"]
                    }
                    ensemble_relations.append(relation)
        
        return ensemble_relations
