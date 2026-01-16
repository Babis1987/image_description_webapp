"""
Face analysis module using DeepFace.

Analyzes emotions, age, and gender from cropped face images.
Based on `emotion_from_face_crop` function from the notebook.
"""

import numpy as np
from typing import Dict, Any, List
from deepface import DeepFace
import cv2
from config import FACE_ANALYSIS



class FaceAnalyzer:
    """
    Face analyzer using DeepFace.
    
    Performs emotion recognition, age estimation, and gender detection
    on cropped face images. 
    """
    
    def __init__(self, actions: List[str] = None):
        """
        Initialize face analyzer.
        
        Args:
            actions: Analysis actions to perform. 
                     If None, uses config default from FACE_ANALYSIS
                     Options: 'emotion', 'age', 'gender', 'race'
        """
        self.actions = actions or FACE_ANALYSIS["actions"]
    
    def analyze_face(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Analyze emotion, age, and gender from a cropped face image.
        
        Args:
            face_bgr: BGR image of cropped face 
        
        Returns:
            Dict with:
            - 'dominant_emotion': str (e.g. 'happy', 'sad', 'angry')
            - 'emotion_scores': dict with probabilities for each emotion
            - 'age': int (estimated age)
            - 'gender': str ('Man' or 'Woman')
            - 'gender_scores': dict with Man/Woman probabilities
        
        Raises:
            ValueError: If face_bgr is empty or invalid
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Face crop is empty or invalid")
        
        # Convert BGR to RGB for DeepFace

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        # IMPORTANT: detector_backend="skip" to avoid re-detecting face
        # Note: img_path can accept numpy array (not just file path)
        analysis = DeepFace.analyze(
            img_path=face_rgb,
            actions=self.actions,
            detector_backend="skip",
            enforce_detection=False
        )
        
        # Handle list or dict response (version compatibility)
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Build result dict with all requested analyses
        result = {
            "dominant_emotion": analysis.get("dominant_emotion", None),
            "emotion_scores": analysis.get("emotion", {}),
        }
        
        # Add age if requested
        if 'age' in self.actions:
            result["age"] = analysis.get("age", None)
        
        # Add gender if requested
        if 'gender' in self.actions:
            result["gender"] = analysis.get("dominant_gender", None)  # 'Man' or 'Woman'
            result["gender_scores"] = analysis.get("gender", {})  # Dict with Man/Woman probabilities
        
        return result
