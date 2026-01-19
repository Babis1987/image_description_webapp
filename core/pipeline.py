"""
Image processing pipeline that integrates all modules:
- Face detection
- Emotion analysis  
- Visualization
- LLM description generation
"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from core.face_detection import FaceDetector
from core.face_analysis import FaceAnalyzer
from core.visualization import ImageVisualizer
from core.description_generator import DescriptionGenerator


logger = logging.getLogger(__name__)


def process_image(
    image: np.ndarray,
    detector: FaceDetector = None,
    analyzer: FaceAnalyzer = None,
    visualizer: ImageVisualizer = None,
    generator: DescriptionGenerator = None,
    generate_description: bool = False,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Process an image through the complete pipeline.

    Args:
        image: BGR image as numpy array
        detector: FaceDetector instance 
        analyzer: FaceAnalyzer instance 
        visualizer: ImageVisualizer instance 
        generator: DescriptionGenerator instance
        generate_description: Whether to generate LLM description
        visualize: Whether to generate annotated image
    
    Returns:
        Dict with:
        - 'success': bool - whether processing succeeded
        - 'faces_detected': int - number of faces found
        - 'faces': list - detailed face data (bbox, emotion, age, gender, position)
        - 'annotated_image': np.ndarray - visualization (if visualize=True)
        - 'description': str - LLM description (if generate_description=True)
        - 'error': str - error message (if success=False)
    
    """
    # Create instances if not provided
    if detector is None:
        detector = FaceDetector()
    if analyzer is None:
        analyzer = FaceAnalyzer()
    if visualizer is None:
        visualizer = ImageVisualizer()
    
    result = {
        "success": False,
        "faces_detected": 0,
        "faces": [],
        "annotated_image": None,
        "description": None,
        "error": None
    }
    
    try:
        # Step 1: Detect faces
        logger.debug("Step 1: Detecting faces...")
        faces_raw = detector.detect_faces(image)
        
        # Filter out low-confidence detections (false positives)
        # DeepFace returns confidence as a float 0-1
        MIN_CONFIDENCE = 0.3  # Minimum confidence threshold
        faces_raw = [
            f for f in faces_raw 
            if f.get('confidence', 0.0) >= MIN_CONFIDENCE
        ]
        
        result["faces_detected"] = len(faces_raw)
        
        if len(faces_raw) == 0:
            logger.info("No faces detected")
            result["success"] = True
            result["description"] = "No faces detected in the image."
            result["annotated_image"] = image.copy() if visualize else None
            return result
        
        logger.info(f"Detected {len(faces_raw)} face(s)")
        
        # Step 2 & 3: Analyze faces and calculate positions
        logger.debug("Step 2-3: Analyzing faces and calculating positions...")
        faces_data = []
        img_height, img_width = image.shape[:2]
        
        for i, face_raw in enumerate(faces_raw, start=1):
            try:
                # Extract bbox
                bbox_dict = face_raw.get("facial_area", {})
                
                # Validate bbox exists and has required keys
                if not all(k in bbox_dict for k in ["x", "y", "w", "h"]):
                    logger.warning(f"Face {i}: Invalid bbox structure, skipping")
                    continue
                
                # Validate bbox values (must be positive and reasonable)
                x = bbox_dict["x"]
                y = bbox_dict["y"]
                w = bbox_dict["w"]
                h = bbox_dict["h"]
                if w <= 0 or h <= 0:
                    logger.warning(f"Face {i}: Invalid bbox dimensions (w={w}, h={h}), skipping")
                    continue
                
                # Validate bbox is within image bounds
                if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                    logger.warning(f"Face {i}: Bbox outside image bounds, skipping")
                    continue
                
                # Crop face for analysis
                face_crop = detector.crop_face_with_margin(image, bbox_dict)
                
                # Analyze face
                analysis = analyzer.analyze_face(face_crop)
                
                # Calculate position
                position = detector.get_face_position(
                    bbox_dict,
                    img_width,
                    img_height
                )
                
                # Build face data
                face_data = {
                    "bbox": (x, y, w, h),
                    "emotion_label": analysis["dominant_emotion"],
                    "emotion_scores": analysis.get("emotion_scores", {}),
                    "age": analysis.get("age"),
                    "gender": analysis.get("gender"),
                    "position": position,
                    "confidence": face_raw.get("confidence", 0)
                }
                
                faces_data.append(face_data)
                logger.debug(f"Face {i}: {analysis['dominant_emotion']}, age~{analysis.get('age')}, {analysis.get('gender')}")
            
            except Exception as e:
                logger.warning(f"Failed to process face {i}: {e}")
                continue
        
        # Update faces_detected to reflect actually processed faces
        result["faces_detected"] = len(faces_data)
        result["faces"] = faces_data
        
        # If no faces were successfully processed, return appropriate message
        if len(faces_data) == 0:
            logger.info("No valid faces found after processing")
            result["success"] = True
            result["description"] = "No faces detected in the image."
            result["annotated_image"] = image.copy() if visualize else None
            return result
        
        # Step 4: Generate visualization
        if visualize:
            logger.debug("Step 4: Generating visualization...")
            result["annotated_image"] = visualizer.draw_results(
                image,
                faces_data
            )
        
        
        # Step 5: Generate LLM description
        if generate_description and generator is not None:
            logger.debug("Step 5: Generating LLM description...")
            try:
                description = generator.generate_description(faces_data)
                result["description"] = description
                logger.info("Description generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate description: {e}")
                result["description"] = f"[Error generating description: {e}]"
        
        result["success"] = True
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        result["error"] = str(e)
        result["annotated_image"] = image.copy() if visualize else None
    
    return result


