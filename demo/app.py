"""Streamlit demo for medical relation extraction."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import torch
from typing import List, Dict, Any
import numpy as np

from src.models import RuleBasedRelationExtractor, TransformerRelationExtractor, EnsembleRelationExtractor, RelationExtractionConfig
from src.data import SyntheticClinicalDataset, DataProcessor
from src.metrics import RelationExtractionMetrics


# Page configuration
st.set_page_config(
    page_title="Medical Relation Extraction Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .relation-highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research demonstration tool only.</strong></p>
    <ul>
        <li>❌ <strong>NOT for clinical use</strong></li>
        <li>❌ <strong>NOT medical advice</strong></li>
        <li>✅ <strong>Requires clinician supervision</strong> for any medical applications</li>
        <li>✅ <strong>For research and education purposes only</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏥 Medical Relation Extraction Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Rule-Based", "Transformer", "Ensemble"],
    help="Choose the type of relation extraction model to use"
)

# De-identification toggle
deidentify = st.sidebar.checkbox(
    "Enable De-identification",
    value=True,
    help="Automatically remove or mask personally identifiable information"
)

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Minimum confidence score for displaying relations"
)

# Load models (with caching)
@st.cache_resource
def load_models():
    """Load all available models."""
    config = RelationExtractionConfig()
    
    try:
        rule_model = RuleBasedRelationExtractor()
        transformer_model = TransformerRelationExtractor(config)
        ensemble_model = EnsembleRelationExtractor(config)
        
        return {
            "rule": rule_model,
            "transformer": transformer_model,
            "ensemble": ensemble_model
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

models = load_models()

if models is None:
    st.error("Failed to load models. Please check the installation.")
    st.stop()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Interactive Demo", "📊 Model Comparison", "📈 Evaluation Metrics", "ℹ️ About"])

with tab1:
    st.header("Interactive Relation Extraction")
    
    # Input options
    input_option = st.radio(
        "Choose input method:",
        ["Enter custom text", "Use sample text", "Generate synthetic text"]
    )
    
    if input_option == "Enter custom text":
        text_input = st.text_area(
            "Enter clinical text:",
            height=150,
            placeholder="Example: The patient was prescribed metformin to manage type 2 diabetes and later developed nausea."
        )
    elif input_option == "Use sample text":
        sample_texts = [
            "The patient was prescribed metformin to manage type 2 diabetes and later developed nausea.",
            "Aspirin was administered to prevent stroke in the elderly patient.",
            "The patient's hypertension was caused by excessive sodium intake.",
            "Chemotherapy treatment worsened the patient's fatigue and caused hair loss.",
            "Insulin therapy was initiated to control blood glucose levels in diabetic patients."
        ]
        
        selected_sample = st.selectbox("Select sample text:", sample_texts)
        text_input = selected_sample
    else:  # Generate synthetic text
        if st.button("Generate New Sample"):
            dataset_generator = SyntheticClinicalDataset()
            sample = dataset_generator.generate_sample()
            text_input = sample.text
        else:
            text_input = "The patient was prescribed metformin to manage type 2 diabetes and later developed nausea."
    
    # Process text
    if text_input and st.button("Extract Relations"):
        with st.spinner("Processing text..."):
            # Apply de-identification if enabled
            if deidentify:
                processor = DataProcessor(deidentify=True)
                processed_text = processor.process_text(text_input)
            else:
                processed_text = text_input
            
            # Extract relations based on selected model
            if model_type == "Rule-Based":
                relations = models["rule"].extract_relations(processed_text)
            elif model_type == "Transformer":
                # For transformer, we need to first extract entities
                entities = models["rule"].extract_entities(processed_text)
                relations = []
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        ent1, _, _, _ = entities[i]
                        ent2, _, _, _ = entities[j]
                        pred = models["transformer"].predict_relation(processed_text, ent1, ent2)
                        if pred["confidence"] >= confidence_threshold:
                            relations.append({
                                "entity1": ent1,
                                "entity2": ent2,
                                "relation": pred["relation"],
                                "confidence": pred["confidence"]
                            })
            else:  # Ensemble
                relations = models["ensemble"].extract_relations(processed_text)
                relations = [r for r in relations if r["confidence"] >= confidence_threshold]
            
            # Display results
            st.subheader("📝 Input Text")
            st.text_area("", processed_text, height=100, disabled=True)
            
            st.subheader("🔗 Extracted Relations")
            
            if relations:
                for i, relation in enumerate(relations):
                    with st.expander(f"Relation {i+1}: {relation['entity1']} → {relation['entity2']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Entity 1:** {relation['entity1']}")
                            st.markdown(f"**Entity 2:** {relation['entity2']}")
                            st.markdown(f"**Relation Type:** {relation['relation']}")
                        
                        with col2:
                            confidence = relation['confidence']
                            st.markdown(f"**Confidence:** {confidence:.3f}")
                            
                            # Confidence bar
                            st.progress(confidence)
                            
                            # Color-coded confidence
                            if confidence >= 0.8:
                                st.success("High Confidence")
                            elif confidence >= 0.6:
                                st.warning("Medium Confidence")
                            else:
                                st.error("Low Confidence")
                
                # Summary statistics
                st.subheader("📊 Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Relations", len(relations))
                
                with col2:
                    avg_confidence = np.mean([r["confidence"] for r in relations])
                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
                
                with col3:
                    relation_types = [r["relation"] for r in relations]
                    most_common = max(set(relation_types), key=relation_types.count) if relation_types else "None"
                    st.metric("Most Common Type", most_common)
                
            else:
                st.warning("No relations found above the confidence threshold.")

with tab2:
    st.header("Model Comparison")
    
    # Test text
    test_text = st.text_area(
        "Test text for comparison:",
        value="The patient was prescribed metformin to manage type 2 diabetes and later developed nausea.",
        height=100
    )
    
    if st.button("Compare Models"):
        with st.spinner("Running comparison..."):
            results = {}
            
            # Rule-based
            rule_relations = models["rule"].extract_relations(test_text)
            results["Rule-Based"] = {
                "relations": rule_relations,
                "count": len(rule_relations),
                "avg_confidence": np.mean([r["confidence"] for r in rule_relations]) if rule_relations else 0
            }
            
            # Transformer
            entities = models["rule"].extract_entities(test_text)
            transformer_relations = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    ent1, _, _, _ = entities[i]
                    ent2, _, _, _ = entities[j]
                    pred = models["transformer"].predict_relation(test_text, ent1, ent2)
                    transformer_relations.append({
                        "entity1": ent1,
                        "entity2": ent2,
                        "relation": pred["relation"],
                        "confidence": pred["confidence"]
                    })
            
            results["Transformer"] = {
                "relations": transformer_relations,
                "count": len(transformer_relations),
                "avg_confidence": np.mean([r["confidence"] for r in transformer_relations]) if transformer_relations else 0
            }
            
            # Ensemble
            ensemble_relations = models["ensemble"].extract_relations(test_text)
            results["Ensemble"] = {
                "relations": ensemble_relations,
                "count": len(ensemble_relations),
                "avg_confidence": np.mean([r["confidence"] for r in ensemble_relations]) if ensemble_relations else 0
            }
            
            # Display comparison
            st.subheader("📊 Comparison Results")
            
            # Metrics comparison
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    "Model": model_name,
                    "Relations Found": result["count"],
                    "Avg Confidence": result["avg_confidence"]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Relations Found", "Average Confidence"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=comparison_df["Model"], y=comparison_df["Relations Found"], name="Relations"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=comparison_df["Model"], y=comparison_df["Avg Confidence"], name="Confidence"),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.subheader("🔍 Detailed Results")
            for model_name, result in results.items():
                with st.expander(f"{model_name} Results"):
                    if result["relations"]:
                        for i, relation in enumerate(result["relations"]):
                            st.markdown(f"**{i+1}.** {relation['entity1']} → {relation['entity2']} ({relation['relation']}) - {relation['confidence']:.3f}")
                    else:
                        st.info("No relations found")

with tab3:
    st.header("Evaluation Metrics")
    
    st.info("This section would show comprehensive evaluation metrics from model training and testing.")
    
    # Placeholder for evaluation metrics
    st.subheader("Model Performance")
    
    # Sample metrics (in a real implementation, these would come from actual evaluation)
    metrics_data = {
        "Model": ["Rule-Based", "Transformer", "Ensemble"],
        "Accuracy": [0.75, 0.82, 0.85],
        "Precision": [0.73, 0.80, 0.83],
        "Recall": [0.77, 0.84, 0.87],
        "F1-Score": [0.75, 0.82, 0.85]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Performance chart
    fig = px.bar(
        metrics_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric",
        title="Model Performance Comparison",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("About This Demo")
    
    st.markdown("""
    ## Medical Relation Extraction
    
    This demo showcases different approaches to extracting relationships between medical entities in clinical text.
    
    ### Models Available:
    
    1. **Rule-Based Extractor**: Uses pattern matching and linguistic rules to identify relations
    2. **Transformer Extractor**: Leverages pre-trained biomedical language models (BioBERT)
    3. **Ensemble Extractor**: Combines both approaches for improved performance
    
    ### Relation Types:
    - **treats**: Drug treats disease
    - **causes**: Symptom/condition causes disease
    - **prevents**: Drug prevents condition
    - **worsens**: Drug worsens symptom
    - **side_effect_of**: Symptom is side effect of drug
    
    ### Features:
    - Interactive text input
    - Real-time relation extraction
    - Confidence scoring
    - De-identification capabilities
    - Model comparison
    - Performance metrics
    
    ### Technical Details:
    - Built with PyTorch and Transformers
    - Uses scispaCy for biomedical NER
    - Implements attention-based relation classification
    - Includes uncertainty quantification
    
    ### Safety Features:
    - Automatic de-identification
    - Confidence thresholds
    - Research-only disclaimer
    - No PHI storage
    """)
    
    st.subheader("🔧 Technical Stack")
    tech_stack = {
        "Framework": "PyTorch, Transformers",
        "NLP": "scispaCy, spaCy",
        "Models": "BioBERT, ClinicalBERT",
        "UI": "Streamlit",
        "Visualization": "Plotly",
        "Data": "Pandas, NumPy"
    }
    
    for tech, description in tech_stack.items():
        st.markdown(f"**{tech}**: {description}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Medical Relation Extraction Demo - Research Use Only | "
    "Not for Clinical Use | Requires Clinician Supervision"
    "</div>",
    unsafe_allow_html=True
)
