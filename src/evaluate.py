"""Evaluation script for medical relation extraction models."""

import argparse
import json
import os
import torch
from typing import Dict, List, Any
import logging

from src.models import RuleBasedRelationExtractor, TransformerRelationExtractor, EnsembleRelationExtractor, RelationExtractionConfig
from src.data import SyntheticClinicalDataset, DataProcessor
from src.metrics import RelationExtractionEvaluator
from src.utils import set_seed, get_device, setup_logging


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Medical Relation Extraction Models")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pt",
                      help="Path to trained model")
    parser.add_argument("--test_data", type=str, default="data/test.json",
                      help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="outputs",
                      help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                      help="Device to use")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging("INFO", os.path.join(args.output_dir, "evaluation.log"))
    
    # Set seed
    set_seed(args.seed)
    
    # Determine device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = RelationExtractionConfig()
    
    # Load test data
    if os.path.exists(args.test_data):
        logger.info(f"Loading test data from {args.test_data}")
        processor = DataProcessor()
        test_samples = processor.load_dataset(args.test_data)
    else:
        logger.info("Generating synthetic test data")
        dataset_generator = SyntheticClinicalDataset()
        test_samples = dataset_generator.generate_dataset(100)
        
        # Save test data
        processor = DataProcessor()
        processor.save_dataset(test_samples, args.test_data)
    
    logger.info(f"Loaded {len(test_samples)} test samples")
    
    # Initialize models
    models = {}
    
    # Rule-based model
    logger.info("Initializing rule-based model...")
    try:
        models["rule_based"] = RuleBasedRelationExtractor()
        logger.info("Rule-based model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize rule-based model: {e}")
        models["rule_based"] = None
    
    # Transformer model
    logger.info("Initializing transformer model...")
    try:
        transformer_model = TransformerRelationExtractor(config)
        if os.path.exists(args.model_path):
            transformer_model.load_state_dict(torch.load(args.model_path, map_location=device))
            logger.info(f"Loaded trained model from {args.model_path}")
        else:
            logger.warning(f"Model file {args.model_path} not found, using untrained model")
        
        transformer_model.to(device)
        models["transformer"] = transformer_model
        logger.info("Transformer model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize transformer model: {e}")
        models["transformer"] = None
    
    # Ensemble model
    logger.info("Initializing ensemble model...")
    try:
        if models["rule_based"] and models["transformer"]:
            ensemble_model = EnsembleRelationExtractor(config)
            ensemble_model.transformer_extractor.load_state_dict(
                torch.load(args.model_path, map_location=device)
            )
            ensemble_model.transformer_extractor.to(device)
            models["ensemble"] = ensemble_model
            logger.info("Ensemble model initialized successfully")
        else:
            logger.warning("Cannot initialize ensemble model due to missing components")
            models["ensemble"] = None
    except Exception as e:
        logger.error(f"Failed to initialize ensemble model: {e}")
        models["ensemble"] = None
    
    # Evaluate models
    evaluator = RelationExtractionEvaluator(config.relation_types)
    results = {}
    
    for model_name, model in models.items():
        if model is None:
            logger.warning(f"Skipping {model_name} due to initialization failure")
            continue
        
        logger.info(f"Evaluating {model_name} model...")
        
        try:
            if model_name == "transformer":
                model_type = "transformer"
            else:
                model_type = "rule_based"
            
            result = evaluator.evaluate_model(model, test_samples, model_type)
            results[model_name] = result
            
            logger.info(f"{model_name} - Accuracy: {result['overall']['accuracy']:.3f}, "
                       f"F1: {result['overall']['f1']:.3f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Create leaderboard
    if results:
        leaderboard = evaluator.create_leaderboard(results)
        logger.info("\n" + leaderboard)
        
        # Save leaderboard
        leaderboard_file = os.path.join(args.output_dir, "leaderboard.txt")
        with open(leaderboard_file, 'w') as f:
            f.write(leaderboard)
        
        logger.info(f"Leaderboard saved to {leaderboard_file}")
    
    # Generate detailed report
    logger.info("Generating detailed evaluation report...")
    
    report = "Medical Relation Extraction Evaluation Report\n"
    report += "=" * 50 + "\n\n"
    
    for model_name, result in results.items():
        if "error" in result:
            report += f"{model_name}: ERROR - {result['error']}\n\n"
            continue
        
        overall = result['overall']
        report += f"{model_name} Model:\n"
        report += f"  Accuracy:  {overall['accuracy']:.3f}\n"
        report += f"  Precision: {overall['precision']:.3f}\n"
        report += f"  Recall:    {overall['recall']:.3f}\n"
        report += f"  F1-Score:  {overall['f1']:.3f}\n"
        report += f"  Support:   {overall['support']}\n\n"
        
        # Per-relation metrics
        if 'per_relation' in result:
            report += "  Per-Relation Performance:\n"
            for relation_type, metrics in result['per_relation'].items():
                report += f"    {relation_type}:\n"
                report += f"      Precision: {metrics['precision']:.3f}\n"
                report += f"      Recall:    {metrics['recall']:.3f}\n"
                report += f"      F1-Score:  {metrics['f1']:.3f}\n"
                report += f"      Support:   {metrics['support']}\n"
            report += "\n"
    
    # Save report
    report_file = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Detailed report saved to {report_file}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
