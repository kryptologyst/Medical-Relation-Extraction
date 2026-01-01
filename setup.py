"""Setup script for Medical Relation Extraction project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medical-relation-extraction",
    version="1.0.0",
    author="Healthcare AI Research Team",
    author_email="research@healthcare-ai.org",
    description="Medical Relation Extraction - Research Demo Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/healthcare-ai/medical-relation-extraction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "pre-commit>=3.3.0",
        ],
        "demo": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
        ],
        "advanced": [
            "wandb>=0.15.0",
            "mlflow>=2.5.0",
            "opacus>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mre-train=src.train:main",
            "mre-demo=demo.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "data/*.json"],
    },
)
