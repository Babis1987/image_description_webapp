"""
Core module for image description bot.

This module contains the core functionality:
- Face detection (face_detection.py)
- Face analysis (face_analysis.py)
- Visualization/drawing results (visualization.py)
- LLM description generation (description_generator.py)
- Pipeline orchestration (pipeline.py)
"""

from .face_detection import FaceDetector
from .face_analysis import FaceAnalyzer
from .visualization import ImageVisualizer
from .description_generator import DescriptionGenerator
from .pipeline import process_image

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "FaceDetector",
    "FaceAnalyzer",
    "ImageVisualizer",
    "DescriptionGenerator",
    
    # Pipeline function
    "process_image",
]
