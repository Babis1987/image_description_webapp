"""
Visualization module for drawing face detection and emotion results.

Provides functions to annotate images with bounding boxes, emotion labels,
and confidence scores. Based on functions from the notebook.
"""

import numpy as np
import cv2
import os
from typing import List, Dict, Any, Optional

from config import EMOTION_COLORS_BGR, VISUALIZATION


class ImageVisualizer:
    """
    Visualizer for face detection and emotion analysis results.
    
    Draws bounding boxes with emotion-based colors and labels.
    """
    
    def __init__(
        self, 
        font_scale: Optional[float] = None, 
        thickness: Optional[int] = None,
        emotion_colors: Optional[Dict[str, tuple]] = None,
        base_resolution: int = 720  # Reference resolution for scaling
    ):
        """
        Initialize visualizer.
        
        Args:
            font_scale: Base scale factor for text labels (if None, uses config default)
            thickness: Base thickness of bounding box lines (if None, uses config default)
            emotion_colors: Optional dict of emotion->BGR color mappings
                          If None, uses config EMOTION_COLORS_BGR
            base_resolution: Reference resolution for dynamic scaling (default 720p)
        """
        self.base_font_scale = font_scale if font_scale is not None else VISUALIZATION["font_scale"]
        self.base_thickness = thickness if thickness is not None else VISUALIZATION["thickness"]
        self.emotion_colors = emotion_colors or EMOTION_COLORS_BGR
        self.base_resolution = base_resolution
    
    def _calculate_scale_factor(self, image_height: int, image_width: int) -> float:
        """
        Calculate scale factor based on image resolution.
        
        Uses the larger dimension to determine scaling.
        
        Args:
            image_height: Height of the image
            image_width: Width of the image
        
        Returns:
            Scale factor relative to base_resolution
        """
        max_dimension = max(image_height, image_width)
        return max_dimension / self.base_resolution
    
    def draw_results(
        self, 
        image_bgr: np.ndarray, 
        results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw bounding boxes and emotion labels on image.
        
        Based on `draw_results` from notebook.
        Box color changes based on detected emotion.
        Thickness and font size scale automatically with image resolution.
        
        Args:
            image_bgr: Original BGR image
            results: List of face detection/analysis results.
                    Each dict should contain:
                    - 'bbox': (x, y, w, h) tuple
                    - 'emotion_label': str (e.g. 'happy', 'sad')
                    - 'emotion_scores': dict with emotion probabilities (optional)
        
        Returns:
            Annotated image with bounding boxes and labels (BGR format)
        """
        img = image_bgr.copy()
        img_height, img_width = img.shape[:2]
        
        # Calculate dynamic scaling based on image resolution
        scale_factor = self._calculate_scale_factor(img_height, img_width)
        
        # Apply scaling to font and thickness (with minimum values)
        font_scale = max(0.6, self.base_font_scale * scale_factor)
        thickness = max(2, int(self.base_thickness * scale_factor))
        
        for r in results:
            x, y, w, h = r["bbox"]
            label = r.get("emotion_label") or "unknown"
            
            # Get confidence score if available
            scores = r.get("emotion_scores", {})
            conf = None
            if isinstance(scores, dict) and label in scores:
                conf = scores[label]
            
            # Format label text
            text = f"{label}" if conf is None else f"{label}: {conf:.2f}"
            
            # Get color based on emotion (case-insensitive)
            color = self.emotion_colors.get(label.lower(), (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
            
            # Calculate text position (with scaled offset)
            text_y = max(int(30 * scale_factor), y - int(10 * scale_factor))
            
            # Draw label text above bounding box
            cv2.putText(
                img, 
                text, 
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                color, 
                thickness
            )
        
        return img


