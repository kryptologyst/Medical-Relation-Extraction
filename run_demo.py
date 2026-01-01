#!/usr/bin/env python3
"""Quick start script for Medical Relation Extraction demo."""

import subprocess
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'spacy',
        'pandas',
        'numpy',
        'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def install_spacy_model():
    """Install required spaCy model."""
    try:
        import spacy
        nlp = spacy.load("en_ner_bc5cdr_md")
        print("spaCy biomedical model already installed")
    except OSError:
        print("Installing spaCy biomedical model...")
        subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_ner_bc5cdr_md"
        ], check=True)
        print("spaCy biomedical model installed successfully")


def run_demo():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found at {demo_path}")
        return False
    
    print("Starting Medical Relation Extraction Demo...")
    print("The demo will open in your web browser.")
    print("Press Ctrl+C to stop the demo.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_path)
        ], check=True)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        return False
    
    return True


def main():
    """Main function."""
    print("Medical Relation Extraction - Quick Start")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Install spaCy model
    install_spacy_model()
    
    # Run demo
    if run_demo():
        print("Demo completed successfully!")
        return 0
    else:
        print("Demo failed to run")
        return 1


if __name__ == "__main__":
    sys.exit(main())
