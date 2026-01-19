"""
Face detection module using DeepFace.
Detects faces in images using multiple backend detectors.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from deepface import DeepFace
import cv2

from config import FACE_DETECTION


class FaceDetector:
    """Face detector using DeepFace with configurable backend (Mistral or Flan-t5)."""
    
    def __init__(
        self, 
        backend: str = None, 
        margin: float = None
    ):
        """
        Initialize face detector.
        
        Args:
            backend: Detection backend (if None, uses config default)
            margin: Margin percentage for face cropping (if None, uses config default)
        """
        self.backend = backend or FACE_DETECTION["backend"]
        self.margin = margin or FACE_DETECTION["margin"]
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and extract faces from an image.
        
        Args:
            image: BGR image as numpy array
        
        Returns:
            List of dicts with:
            - 'facial_area': bounding box dict {x, y, w, h}
            - 'face': cropped (and implicitly aligned) face
            - 'confidence': detection confidence score
        """
        # Convert BGR to RGB for DeepFace
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = DeepFace.extract_faces(
            img_rgb,
            detector_backend=self.backend,
            enforce_detection=False,
            align=True
        )
        
        return faces
    
    def crop_face_with_margin(
        self, 
        image: np.ndarray, 
        bbox: Dict[str, int]
    ) -> np.ndarray:
        """
        Crop face from image with margin expansion.
        
        Args:
            image: BGR image
            bbox: Bounding box dict with keys {x, y, w, h}
        
        Returns:
            Cropped face as BGR numpy array
        """
        h_img, w_img = image.shape[:2]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        
        # Calculate margin
        dx = int(w * self.margin)
        dy = int(h * self.margin)
        
        # Apply margin with boundary checks
        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(w_img, x + w + dx)
        y2 = min(h_img, y + h + dy)
        
        face_crop = image[y1:y2, x1:x2].copy()
        return face_crop
    
    @staticmethod
    def get_face_position(
        bbox: Dict[str, int],
        img_width: int,
        img_height: int
    ) -> Dict[str, Any]:
        """
        Calculate face position in image (for natural language descriptions).
        
        Args:
            bbox: Tuple (x, y, w, h)
            img_width: Image width
            img_height: Image height
        
        Returns:
            Dict with:
            - 'horizontal': 'left', 'center', 'right'
            - 'vertical': 'top', 'middle', 'bottom'
            - 'position': combined string (e.g. 'top-left', 'center')
            - 'center_x': normalized x coordinate (0-1)
            - 'center_y': normalized y coordinate (0-1)
        """
        x = bbox["x"]
        y = bbox["y"] 
        w = bbox["w"] 
        h = bbox["h"]
        
        # Calculate center point
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Normalize coordinates (0-1)
        norm_x = center_x / img_width
        norm_y = center_y / img_height
        
        # Determine horizontal position
        if norm_x < 0.33:
            horizontal = "left"
        elif norm_x > 0.67:
            horizontal = "right"
        else:
            horizontal = "center"
        
        # Determine vertical position
        if norm_y < 0.33:
            vertical = "top"
        elif norm_y > 0.67:
            vertical = "bottom"
        else:
            vertical = "middle"
        
        # Combined position
        if horizontal == "center" and vertical == "middle":
            position = "center"
        else:
            position = f"{vertical}-{horizontal}"
        
        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "position": position,
            "center_x": round(norm_x, 3),
            "center_y": round(norm_y, 3)
        }