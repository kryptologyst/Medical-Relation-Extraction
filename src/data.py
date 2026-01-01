"""Data utilities and synthetic dataset generation for medical relation extraction."""

import json
import random
from typing import List, Dict, Any, Tuple
import pandas as pd
from dataclasses import dataclass
import re


@dataclass
class ClinicalTextSample:
    """Represents a clinical text sample with annotations."""
    text: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    deidentified_text: str = ""


class SyntheticClinicalDataset:
    """Generates synthetic clinical text for relation extraction."""
    
    def __init__(self):
        """Initialize the synthetic dataset generator."""
        # Medical entities
        self.diseases = [
            "diabetes", "hypertension", "asthma", "pneumonia", "depression",
            "anxiety", "arthritis", "cancer", "stroke", "heart disease",
            "type 2 diabetes", "chronic kidney disease", "COPD", "migraine"
        ]
        
        self.drugs = [
            "metformin", "lisinopril", "albuterol", "amoxicillin", "sertraline",
            "ibuprofen", "chemotherapy", "warfarin", "aspirin", "insulin",
            "prednisone", "omeprazole", "atorvastatin", "levothyroxine"
        ]
        
        self.symptoms = [
            "nausea", "headache", "fatigue", "chest pain", "shortness of breath",
            "dizziness", "fever", "cough", "rash", "joint pain",
            "weight loss", "insomnia", "anxiety", "depression"
        ]
        
        # Relation templates
        self.treatment_templates = [
            "The patient was prescribed {drug} to manage {disease}.",
            "{drug} was administered to treat {disease}.",
            "Treatment with {drug} was initiated for {disease}.",
            "The patient is taking {drug} for {disease} management."
        ]
        
        self.side_effect_templates = [
            "The patient developed {symptom} after starting {drug}.",
            "{drug} caused {symptom} in the patient.",
            "The patient experienced {symptom} as a side effect of {drug}.",
            "{symptom} appeared following {drug} administration."
        ]
        
        self.causation_templates = [
            "{disease} was caused by {symptom}.",
            "The patient's {disease} resulted from {symptom}.",
            "{symptom} led to the development of {disease}.",
            "{disease} was triggered by {symptom}."
        ]
    
    def generate_sample(self) -> ClinicalTextSample:
        """Generate a single synthetic clinical text sample.
        
        Returns:
            ClinicalTextSample with text, entities, and relations.
        """
        # Choose relation type
        relation_type = random.choice(["treatment", "side_effect", "causation"])
        
        if relation_type == "treatment":
            drug = random.choice(self.drugs)
            disease = random.choice(self.diseases)
            template = random.choice(self.treatment_templates)
            text = template.format(drug=drug, disease=disease)
            
            entities = [
                {"text": drug, "label": "CHEMICAL", "start": text.find(drug), "end": text.find(drug) + len(drug)},
                {"text": disease, "label": "DISEASE", "start": text.find(disease), "end": text.find(disease) + len(disease)}
            ]
            
            relations = [
                {
                    "entity1": drug,
                    "entity2": disease,
                    "relation": "treats",
                    "confidence": 0.9
                }
            ]
            
        elif relation_type == "side_effect":
            drug = random.choice(self.drugs)
            symptom = random.choice(self.symptoms)
            template = random.choice(self.side_effect_templates)
            text = template.format(drug=drug, symptom=symptom)
            
            entities = [
                {"text": drug, "label": "CHEMICAL", "start": text.find(drug), "end": text.find(drug) + len(drug)},
                {"text": symptom, "label": "SYMPTOM", "start": text.find(symptom), "end": text.find(symptom) + len(symptom)}
            ]
            
            relations = [
                {
                    "entity1": drug,
                    "entity2": symptom,
                    "relation": "side_effect_of",
                    "confidence": 0.9
                }
            ]
            
        else:  # causation
            disease = random.choice(self.diseases)
            symptom = random.choice(self.symptoms)
            template = random.choice(self.causation_templates)
            text = template.format(disease=disease, symptom=symptom)
            
            entities = [
                {"text": disease, "label": "DISEASE", "start": text.find(disease), "end": text.find(disease) + len(disease)},
                {"text": symptom, "label": "SYMPTOM", "start": text.find(symptom), "end": text.find(symptom) + len(symptom)}
            ]
            
            relations = [
                {
                    "entity1": symptom,
                    "entity2": disease,
                    "relation": "causes",
                    "confidence": 0.9
                }
            ]
        
        return ClinicalTextSample(
            text=text,
            entities=entities,
            relations=relations,
            deidentified_text=self.deidentify_text(text)
        )
    
    def generate_dataset(self, num_samples: int = 100) -> List[ClinicalTextSample]:
        """Generate a synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate.
            
        Returns:
            List of ClinicalTextSample objects.
        """
        samples = []
        for _ in range(num_samples):
            samples.append(self.generate_sample())
        return samples
    
    def deidentify_text(self, text: str) -> str:
        """Basic de-identification of clinical text.
        
        Args:
            text: Input clinical text.
            
        Returns:
            De-identified text with PHI replaced.
        """
        # Replace common PHI patterns
        deidentified = text
        
        # Replace names (simple pattern)
        deidentified = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PATIENT_NAME]', deidentified)
        
        # Replace dates
        deidentified = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]', deidentified)
        deidentified = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]', deidentified)
        
        # Replace phone numbers
        deidentified = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', deidentified)
        
        # Replace SSN patterns
        deidentified = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', deidentified)
        
        return deidentified


class DataProcessor:
    """Processes clinical text data for relation extraction."""
    
    def __init__(self, deidentify: bool = True):
        """Initialize the data processor.
        
        Args:
            deidentify: Whether to apply de-identification to text.
        """
        self.deidentify = deidentify
        self.deid_patterns = {
            'patient_id': r'\b[A-Z]{2,3}\d{4,6}\b',
            'mrn': r'\bMRN:\s*\d+\b',
            'date': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
    
    def process_text(self, text: str) -> str:
        """Process clinical text with optional de-identification.
        
        Args:
            text: Input clinical text.
            
        Returns:
            Processed text.
        """
        processed_text = text.strip()
        
        if self.deidentify:
            for pattern_name, pattern in self.deid_patterns.items():
                processed_text = re.sub(pattern, f'[{pattern_name.upper()}]', processed_text)
        
        return processed_text
    
    def create_relation_dataset(self, samples: List[ClinicalTextSample]) -> pd.DataFrame:
        """Create a pandas DataFrame for relation extraction training.
        
        Args:
            samples: List of ClinicalTextSample objects.
            
        Returns:
            DataFrame with text, entity pairs, and relations.
        """
        data = []
        
        for sample in samples:
            text = self.process_text(sample.text)
            
            for relation in sample.relations:
                data.append({
                    'text': text,
                    'entity1': relation['entity1'],
                    'entity2': relation['entity2'],
                    'relation': relation['relation'],
                    'confidence': relation['confidence']
                })
        
        return pd.DataFrame(data)
    
    def save_dataset(self, samples: List[ClinicalTextSample], filepath: str):
        """Save dataset to JSON file.
        
        Args:
            samples: List of ClinicalTextSample objects.
            filepath: Path to save the dataset.
        """
        data = []
        for sample in samples:
            data.append({
                'text': sample.text,
                'entities': sample.entities,
                'relations': sample.relations,
                'deidentified_text': sample.deidentified_text
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_dataset(self, filepath: str) -> List[ClinicalTextSample]:
        """Load dataset from JSON file.
        
        Args:
            filepath: Path to the dataset file.
            
        Returns:
            List of ClinicalTextSample objects.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = ClinicalTextSample(
                text=item['text'],
                entities=item['entities'],
                relations=item['relations'],
                deidentified_text=item.get('deidentified_text', '')
            )
            samples.append(sample)
        
        return samples
