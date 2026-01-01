"""Utility functions for medical relation extraction."""

import os
import json
import random
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms (may impact performance)
    torch.use_deterministic_algorithms(True)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu").
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level.
        log_file: Optional log file path.
        
    Returns:
        Configured logger.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save.
        filepath: Path to save file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """Load data from JSON file.
    
    Args:
        filepath: Path to JSON file.
        
    Returns:
        Loaded data.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def format_confidence(confidence: float) -> str:
    """Format confidence score for display.
    
    Args:
        confidence: Confidence score (0-1).
        
    Returns:
        Formatted confidence string.
    """
    return f"{confidence:.3f}"


def get_confidence_color(confidence: float) -> str:
    """Get color for confidence score.
    
    Args:
        confidence: Confidence score (0-1).
        
    Returns:
        Color name for display.
    """
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"


def validate_text_input(text: str, max_length: int = 1000) -> bool:
    """Validate text input for processing.
    
    Args:
        text: Input text to validate.
        max_length: Maximum allowed text length.
        
    Returns:
        True if text is valid, False otherwise.
    """
    if not text or not text.strip():
        return False
    
    if len(text) > max_length:
        return False
    
    # Check for potentially harmful content
    harmful_patterns = ['<script', 'javascript:', 'data:']
    text_lower = text.lower()
    for pattern in harmful_patterns:
        if pattern in text_lower:
            return False
    
    return True


def sanitize_text(text: str) -> str:
    """Sanitize text input for safe processing.
    
    Args:
        text: Input text to sanitize.
        
    Returns:
        Sanitized text.
    """
    # Remove HTML tags
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def calculate_relation_distance(entity1_pos: tuple, entity2_pos: tuple) -> int:
    """Calculate distance between two entities in text.
    
    Args:
        entity1_pos: (start, end) position of first entity.
        entity2_pos: (start, end) position of second entity.
        
    Returns:
        Distance between entities in characters.
    """
    start1, end1 = entity1_pos
    start2, end2 = entity2_pos
    
    if end1 <= start2:
        return start2 - end1
    elif end2 <= start1:
        return start1 - end2
    else:
        return 0  # Overlapping entities


def extract_entity_context(text: str, entity_pos: tuple, context_window: int = 50) -> str:
    """Extract context around an entity.
    
    Args:
        text: Input text.
        entity_pos: (start, end) position of entity.
        context_window: Number of characters to include on each side.
        
    Returns:
        Context string around the entity.
    """
    start, end = entity_pos
    context_start = max(0, start - context_window)
    context_end = min(len(text), end + context_window)
    
    return text[context_start:context_end]


def format_relation_output(relation: Dict[str, Any]) -> str:
    """Format relation for display.
    
    Args:
        relation: Relation dictionary.
        
    Returns:
        Formatted relation string.
    """
    entity1 = relation.get('entity1', '')
    entity2 = relation.get('entity2', '')
    relation_type = relation.get('relation', '')
    confidence = relation.get('confidence', 0.0)
    
    return f"{entity1} → {entity2} ({relation_type}, {confidence:.3f})"


def create_relation_summary(relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics for relations.
    
    Args:
        relations: List of relation dictionaries.
        
    Returns:
        Summary statistics dictionary.
    """
    if not relations:
        return {
            'total_relations': 0,
            'relation_types': {},
            'avg_confidence': 0.0,
            'high_confidence_count': 0
        }
    
    relation_types = [r.get('relation', 'unknown') for r in relations]
    confidences = [r.get('confidence', 0.0) for r in relations]
    
    type_counts = {}
    for rel_type in relation_types:
        type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
    
    return {
        'total_relations': len(relations),
        'relation_types': type_counts,
        'avg_confidence': np.mean(confidences),
        'high_confidence_count': sum(1 for c in confidences if c >= 0.8),
        'max_confidence': max(confidences),
        'min_confidence': min(confidences)
    }


def check_model_availability(model_name: str) -> bool:
    """Check if a model is available for loading.
    
    Args:
        model_name: Name of the model to check.
        
    Returns:
        True if model is available, False otherwise.
    """
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(model_name)
        return True
    except Exception:
        return False


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model.
    
    Args:
        model_name: Name of the model.
        
    Returns:
        Model information dictionary.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        return {
            'name': model_name,
            'available': True,
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'num_hidden_layers': config.num_hidden_layers
        }
    except Exception as e:
        return {
            'name': model_name,
            'available': False,
            'error': str(e)
        }


def benchmark_model_performance(model, test_data: List[Any], num_runs: int = 5) -> Dict[str, float]:
    """Benchmark model performance.
    
    Args:
        model: Model to benchmark.
        test_data: Test data for benchmarking.
        num_runs: Number of runs for averaging.
        
    Returns:
        Performance metrics dictionary.
    """
    import time
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        # Run model inference
        for sample in test_data:
            if hasattr(model, 'extract_relations'):
                model.extract_relations(sample.text)
            elif hasattr(model, 'predict_relation'):
                # For transformer models, we need entity pairs
                entities = model.rule_extractor.extract_entities(sample.text)
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        ent1, _, _, _ = entities[i]
                        ent2, _, _, _ = entities[j]
                        model.predict_relation(sample.text, ent1, ent2)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'avg_time_per_sample': np.mean(times) / len(test_data),
        'total_time': np.mean(times),
        'std_time': np.std(times),
        'samples_per_second': len(test_data) / np.mean(times)
    }
