"""Training script for medical relation extraction models."""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import argparse
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

from src.models import TransformerRelationExtractor, RelationExtractionConfig
from src.data import SyntheticClinicalDataset, DataProcessor, ClinicalTextSample
from src.metrics import RelationExtractionEvaluator


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RelationDataset(Dataset):
    """Dataset class for relation extraction training."""
    
    def __init__(self, samples: List[ClinicalTextSample], tokenizer, config: RelationExtractionConfig):
        """Initialize the dataset.
        
        Args:
            samples: List of ClinicalTextSample objects.
            tokenizer: HuggingFace tokenizer.
            config: Model configuration.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config
        self.relation_types = ["treats", "causes", "prevents", "worsens", "side_effect_of"]
        
        # Create training examples
        self.examples = self._create_examples()
    
    def _create_examples(self) -> List[Dict[str, Any]]:
        """Create training examples from samples."""
        examples = []
        
        for sample in self.samples:
            for relation in sample.relations:
                # Create positive example
                marked_text = f"[E1]{relation['entity1']}[/E1] {sample.text} [E2]{relation['entity2']}[/E2]"
                relation_idx = self.relation_types.index(relation['relation'])
                
                examples.append({
                    'text': marked_text,
                    'relation': relation_idx,
                    'entity1': relation['entity1'],
                    'entity2': relation['entity2']
                })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize
        inputs = self.tokenizer(
            example['text'],
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(example['relation'], dtype=torch.long)
        }


def train_model(model: TransformerRelationExtractor,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: RelationExtractionConfig,
                device: torch.device,
                output_dir: str) -> Dict[str, Any]:
    """Train the transformer model.
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Model configuration.
        device: Device to train on.
        output_dir: Directory to save checkpoints.
        
    Returns:
        Training history dictionary.
    """
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    return history


def evaluate_model(model: TransformerRelationExtractor,
                   test_samples: List[ClinicalTextSample],
                   config: RelationExtractionConfig,
                   device: torch.device) -> Dict[str, Any]:
    """Evaluate the trained model.
    
    Args:
        model: The trained model.
        test_samples: Test samples.
        config: Model configuration.
        device: Device to evaluate on.
        
    Returns:
        Evaluation results.
    """
    model.eval()
    evaluator = RelationExtractionEvaluator(config.relation_types)
    
    results = evaluator.evaluate_model(model, test_samples, "transformer")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Medical Relation Extraction Model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                      help="Output directory for models and logs")
    parser.add_argument("--data_dir", type=str, default="data",
                      help="Directory containing training data")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                      help="Device to use (cuda, mps, cpu, auto)")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = RelationExtractionConfig()
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset...")
    dataset_generator = SyntheticClinicalDataset()
    train_samples = dataset_generator.generate_dataset(800)
    val_samples = dataset_generator.generate_dataset(100)
    test_samples = dataset_generator.generate_dataset(100)
    
    # Save datasets
    data_processor = DataProcessor()
    data_processor.save_dataset(train_samples, os.path.join(args.data_dir, "train.json"))
    data_processor.save_dataset(val_samples, os.path.join(args.data_dir, "val.json"))
    data_processor.save_dataset(test_samples, os.path.join(args.data_dir, "test.json"))
    
    # Initialize model
    logger.info("Initializing model...")
    model = TransformerRelationExtractor(config)
    
    # Create datasets
    train_dataset = RelationDataset(train_samples, model.tokenizer, config)
    val_dataset = RelationDataset(val_samples, model.tokenizer, config)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train model
    logger.info("Starting training...")
    history = train_model(model, train_loader, val_loader, config, device, args.output_dir)
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = evaluate_model(model, test_samples, config, device)
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Print results
    logger.info("Training completed!")
    logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    logger.info(f"Test accuracy: {eval_results['overall']['accuracy']:.4f}")
    logger.info(f"Test F1-score: {eval_results['overall']['f1']:.4f}")


if __name__ == "__main__":
    main()
