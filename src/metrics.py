"""Evaluation metrics and utilities for medical relation extraction."""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class RelationExtractionMetrics:
    """Metrics for evaluating relation extraction performance."""
    
    def __init__(self, relation_types: List[str]):
        """Initialize metrics calculator.
        
        Args:
            relation_types: List of relation types to evaluate.
        """
        self.relation_types = relation_types
        self.results = defaultdict(list)
    
    def add_prediction(self, 
                      true_relation: str, 
                      predicted_relation: str, 
                      confidence: float,
                      entity1: str = "",
                      entity2: str = ""):
        """Add a prediction for evaluation.
        
        Args:
            true_relation: Ground truth relation type.
            predicted_relation: Predicted relation type.
            confidence: Prediction confidence score.
            entity1: First entity in the relation.
            entity2: Second entity in the relation.
        """
        self.results['true_relations'].append(true_relation)
        self.results['predicted_relations'].append(predicted_relation)
        self.results['confidences'].append(confidence)
        self.results['entity1'].append(entity1)
        self.results['entity2'].append(entity2)
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary containing various evaluation metrics.
        """
        if not self.results['true_relations']:
            return {"error": "No predictions to evaluate"}
        
        true_relations = np.array(self.results['true_relations'])
        predicted_relations = np.array(self.results['predicted_relations'])
        confidences = np.array(self.results['confidences'])
        
        # Overall metrics
        accuracy = accuracy_score(true_relations, predicted_relations)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_relations, predicted_relations, average='weighted'
        )
        
        # Per-relation metrics
        per_relation_metrics = {}
        for relation_type in self.relation_types:
            if relation_type in true_relations:
                rel_precision, rel_recall, rel_f1, rel_support = precision_recall_fscore_support(
                    true_relations, predicted_relations, labels=[relation_type], average='binary'
                )
                per_relation_metrics[relation_type] = {
                    'precision': rel_precision[0] if len(rel_precision) > 0 else 0.0,
                    'recall': rel_recall[0] if len(rel_recall) > 0 else 0.0,
                    'f1': rel_f1[0] if len(rel_f1) > 0 else 0.0,
                    'support': rel_support[0] if len(rel_support) > 0 else 0
                }
        
        # Confidence analysis
        correct_predictions = (true_relations == predicted_relations)
        avg_confidence_correct = np.mean(confidences[correct_predictions]) if np.any(correct_predictions) else 0.0
        avg_confidence_incorrect = np.mean(confidences[~correct_predictions]) if np.any(~correct_predictions) else 0.0
        
        # Confusion matrix
        cm = confusion_matrix(true_relations, predicted_relations, labels=self.relation_types)
        
        return {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': len(true_relations)
            },
            'per_relation': per_relation_metrics,
            'confidence_analysis': {
                'avg_confidence_correct': avg_confidence_correct,
                'avg_confidence_incorrect': avg_confidence_incorrect,
                'confidence_gap': avg_confidence_correct - avg_confidence_incorrect
            },
            'confusion_matrix': cm.tolist(),
            'relation_types': self.relation_types
        }
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot confusion matrix.
        
        Args:
            save_path: Optional path to save the plot.
        """
        metrics = self.calculate_metrics()
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.relation_types,
                   yticklabels=self.relation_types)
        plt.title('Relation Extraction Confusion Matrix')
        plt.xlabel('Predicted Relation')
        plt.ylabel('True Relation')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_distribution(self, save_path: str = None):
        """Plot confidence score distribution.
        
        Args:
            save_path: Optional path to save the plot.
        """
        confidences = np.array(self.results['confidences'])
        correct_predictions = np.array(self.results['true_relations']) == np.array(self.results['predicted_relations'])
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(confidences[correct_predictions], bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(confidences[~correct_predictions], bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Prediction Correctness')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot([confidences[correct_predictions], confidences[~correct_predictions]], 
                   labels=['Correct', 'Incorrect'])
        plt.ylabel('Confidence Score')
        plt.title('Confidence Distribution Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a detailed evaluation report.
        
        Returns:
            Formatted string report.
        """
        metrics = self.calculate_metrics()
        
        report = "Medical Relation Extraction Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall metrics
        overall = metrics['overall']
        report += f"Overall Performance:\n"
        report += f"  Accuracy:  {overall['accuracy']:.3f}\n"
        report += f"  Precision: {overall['precision']:.3f}\n"
        report += f"  Recall:    {overall['recall']:.3f}\n"
        report += f"  F1-Score:  {overall['f1']:.3f}\n"
        report += f"  Support:   {overall['support']}\n\n"
        
        # Per-relation metrics
        report += "Per-Relation Performance:\n"
        for relation_type, rel_metrics in metrics['per_relation'].items():
            report += f"  {relation_type}:\n"
            report += f"    Precision: {rel_metrics['precision']:.3f}\n"
            report += f"    Recall:    {rel_metrics['recall']:.3f}\n"
            report += f"    F1-Score:  {rel_metrics['f1']:.3f}\n"
            report += f"    Support:   {rel_metrics['support']}\n"
        
        # Confidence analysis
        conf_analysis = metrics['confidence_analysis']
        report += f"\nConfidence Analysis:\n"
        report += f"  Average confidence (correct):   {conf_analysis['avg_confidence_correct']:.3f}\n"
        report += f"  Average confidence (incorrect): {conf_analysis['avg_confidence_incorrect']:.3f}\n"
        report += f"  Confidence gap:                {conf_analysis['confidence_gap']:.3f}\n"
        
        return report


class RelationExtractionEvaluator:
    """Comprehensive evaluator for relation extraction models."""
    
    def __init__(self, relation_types: List[str]):
        """Initialize the evaluator.
        
        Args:
            relation_types: List of relation types to evaluate.
        """
        self.relation_types = relation_types
        self.metrics = RelationExtractionMetrics(relation_types)
    
    def evaluate_model(self, 
                     model, 
                     test_samples: List[Any],
                     model_type: str = "rule_based") -> Dict[str, Any]:
        """Evaluate a relation extraction model.
        
        Args:
            model: The model to evaluate.
            test_samples: List of test samples.
            model_type: Type of model ("rule_based", "transformer", "ensemble").
            
        Returns:
            Evaluation results dictionary.
        """
        self.metrics = RelationExtractionMetrics(self.relation_types)
        
        for sample in test_samples:
            if model_type == "rule_based":
                predictions = model.extract_relations(sample.text)
            elif model_type == "transformer":
                # For transformer, we need entity pairs
                predictions = []
                for relation in sample.relations:
                    pred = model.predict_relation(
                        sample.text, relation['entity1'], relation['entity2']
                    )
                    predictions.append({
                        'entity1': relation['entity1'],
                        'entity2': relation['entity2'],
                        'relation': pred['relation'],
                        'confidence': pred['confidence']
                    })
            else:  # ensemble
                predictions = model.extract_relations(sample.text)
            
            # Add predictions to metrics
            for true_relation in sample.relations:
                found_match = False
                for pred in predictions:
                    if (pred['entity1'] == true_relation['entity1'] and 
                        pred['entity2'] == true_relation['entity2']) or \
                       (pred['entity1'] == true_relation['entity2'] and 
                        pred['entity2'] == true_relation['entity1']):
                        
                        self.metrics.add_prediction(
                            true_relation['relation'],
                            pred['relation'],
                            pred['confidence'],
                            pred['entity1'],
                            pred['entity2']
                        )
                        found_match = True
                        break
                
                if not found_match:
                    # No prediction found - treat as "no_relation"
                    self.metrics.add_prediction(
                        true_relation['relation'],
                        "no_relation",
                        0.0,
                        true_relation['entity1'],
                        true_relation['entity2']
                    )
        
        return self.metrics.calculate_metrics()
    
    def create_leaderboard(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Create a leaderboard comparing different models.
        
        Args:
            results: Dictionary with model names as keys and metrics as values.
            
        Returns:
            Formatted leaderboard string.
        """
        leaderboard = "Medical Relation Extraction Leaderboard\n"
        leaderboard += "=" * 50 + "\n\n"
        
        # Sort models by F1 score
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1]['overall']['f1'], 
                             reverse=True)
        
        leaderboard += f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n"
        leaderboard += "-" * 60 + "\n"
        
        for model_name, metrics in sorted_models:
            overall = metrics['overall']
            leaderboard += f"{model_name:<20} {overall['accuracy']:<10.3f} {overall['precision']:<10.3f} {overall['recall']:<10.3f} {overall['f1']:<10.3f}\n"
        
        return leaderboard
