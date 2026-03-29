"""
Configuration management for the Streamlit Diabetes Prediction application.
"""
import os
from pathlib import Path
from typing import Dict, Tuple

# Directory containing this file (app root when deployed: repo root on Streamlit Cloud)
_PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_model_path() -> str:
    """
    Resolve the pickle path for local runs, Docker, and Streamlit Cloud.

    - If MODEL_PATH is unset: ``<project>/best_diabetes_model.pkl`` (next to config.py).
    - If MODEL_PATH is absolute: use as-is.
    - If MODEL_PATH is relative: resolve against the project root (not the process cwd),
      so ``streamlit run path/to/app.py`` from another folder still works.
    """
    raw = os.getenv("MODEL_PATH", "").strip()
    if raw:
        p = Path(raw)
        if p.is_absolute():
            return str(p)
        return str((_PROJECT_ROOT / p).resolve())
    return str(_PROJECT_ROOT / "best_diabetes_model.pkl")


class Config:
    """Application configuration loaded from environment variables."""
    
    # Model configuration (resolved at import time)
    MODEL_PATH: str = _resolve_model_path()
    
    # Streamlit configuration
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    
    # Feature information
    FEATURE_INFO: Dict[str, Dict[str, any]] = {
        'Age': {
            'description': 'Age in years',
            'min': 0,
            'max': 120,
            'default': 50.0,
            'unit': 'years'
        },
        'Sex': {
            'description': 'Gender (0 = female, 1 = male)',
            'min': 0,
            'max': 1,
            'default': 0.0,
            'unit': ''
        },
        'BMI': {
            'description': 'Body Mass Index',
            'min': 10,
            'max': 50,
            'default': 25.0,
            'unit': 'kg/m²'
        },
        'BP': {
            'description': 'Average blood pressure',
            'min': 50,
            'max': 150,
            'default': 75.0,
            'unit': 'mm Hg'
        },
        'S1': {
            'description': 'Total cholesterol',
            'min': 100,
            'max': 400,
            'default': 200.0,
            'unit': 'mg/dl'
        },
        'S2': {
            'description': 'Low-density lipoproteins (LDL)',
            'min': 50,
            'max': 300,
            'default': 150.0,
            'unit': 'mg/dl'
        },
        'S3': {
            'description': 'High-density lipoproteins (HDL)',
            'min': 20,
            'max': 100,
            'default': 45.0,
            'unit': 'mg/dl'
        },
        'S4': {
            'description': 'Total cholesterol / HDL',
            'min': 1,
            'max': 10,
            'default': 5.0,
            'unit': 'ratio'
        },
        'S5': {
            'description': 'Log of serum triglycerides level',
            'min': 2,
            'max': 7,
            'default': 4.5,
            'unit': 'log(mg/dl)'
        },
        'S6': {
            'description': 'Blood sugar level',
            'min': 50,
            'max': 200,
            'default': 85.0,
            'unit': 'mg/dl'
        }
    }
    
    # Model performance metrics (from training)
    MODEL_METRICS: Dict[str, float] = {
        'r2_score': 0.4609,
        'mape': 37.6958
    }
    
    @classmethod
    def get_feature_names(cls) -> list:
        """Get list of feature names in order."""
        return list(cls.FEATURE_INFO.keys())
    
    @classmethod
    def get_feature_defaults(cls) -> list:
        """Get default values for all features."""
        return [info['default'] for info in cls.FEATURE_INFO.values()]
    
    @classmethod
    def get_feature_range(cls, feature_name: str) -> Tuple[float, float]:
        """Get min and max range for a feature."""
        info = cls.FEATURE_INFO[feature_name]
        return (info['min'], info['max'])
    
    @classmethod
    def validate_feature(cls, feature_name: str, value: float) -> bool:
        """Validate if a feature value is within acceptable range."""
        info = cls.FEATURE_INFO[feature_name]
        return info['min'] <= value <= info['max']





