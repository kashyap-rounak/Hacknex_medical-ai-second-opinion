"""
Application Configuration for MedSight AI
Supports environment variables and defaults for Hackathon MVP demo.
"""

import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "MedSight AI"
    APP_VERSION: str = "1.0.0"
    MODE: Literal["demo", "production"] = os.getenv("MODE", "demo")
    
    # Model device selection: auto, cuda, mps, cpu
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "auto")
    
    # Hallucination Gate Thresholds
    VISUAL_THRESHOLD: float = float(os.getenv("VISUAL_THRESHOLD", "0.65"))
    TEXT_THRESHOLD: float = float(os.getenv("TEXT_THRESHOLD", "0.70"))
    
    # Confidence Calibration Bands
    CONFIDENCE_HIGH: float = float(os.getenv("CONFIDENCE_HIGH", "0.80"))
    CONFIDENCE_MODERATE: float = float(os.getenv("CONFIDENCE_MODERATE", "0.60"))
    
    # Feature Flags
    ENABLE_CT: bool = os.getenv("ENABLE_CT", "true").lower() == "true"
    ENABLE_SEGMENTATION: bool = os.getenv("ENABLE_SEGMENTATION", "true").lower() == "true"
    ENABLE_HALLUCINATION_GATE: bool = os.getenv("ENABLE_HALLUCINATION_GATE", "true").lower() == "true"
    PHI_ANONYMIZATION: bool = os.getenv("PHI_ANONYMIZATION", "true").lower() == "true"
    
    # Network Ports
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    UI_PORT: int = int(os.getenv("UI_PORT", "8501"))
    
    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to the best available compute device."""
        if self.MODEL_DEVICE != "auto":
            return self.MODEL_DEVICE
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

settings = Settings()