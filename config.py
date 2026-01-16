"""
Configuration module for Image Description Bot.
Loads settings from .env file with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = INSTANCE_DIR / os.getenv("UPLOAD_DIR", "uploads")
ORIGINAL_DIR = UPLOAD_DIR / "original"
PROCESSED_DIR = UPLOAD_DIR / "processed"
LOG_DIR = INSTANCE_DIR / "logs"

# Create directories if they don't exist
for directory in [INSTANCE_DIR, UPLOAD_DIR, ORIGINAL_DIR, PROCESSED_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Flask Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-for-academic-project")
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")

# Upload settings
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
UPLOAD_FOLDER = str(UPLOAD_DIR)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Face Detection Settings
FACE_DETECTION = {
    "backend": os.getenv("FACE_DETECTION_BACKEND", "mediapipe"),
    "margin": float(os.getenv("FACE_DETECTION_MARGIN", "0.20")),
}


# Face Analysis Settings
FACE_ANALYSIS = {
    "actions": os.getenv("FACE_ANALYSIS_ACTIONS", "emotion,age,gender").split(","),
}


# Model Configuration
MODEL_CONFIG = {
    "model_type": os.getenv("LLM_MODEL_TYPE", "mistral").lower(),  # 'mistral' or 'flan-t5'
    "mistral_model_id": os.getenv("MISTRAL_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"),
    "flan_t5_model_id": os.getenv("FLAN_T5_MODEL_ID", "google/flan-t5-base"),
    "use_gpu": os.getenv("USE_GPU", "True").lower() in ("true", "1", "yes"),
    "use_4bit_quantization": os.getenv("USE_4BIT_QUANTIZATION", "True").lower() in ("true", "1", "yes"),
}


# LLM Generation Settings
LLM_CONFIG = {
    "enable_description": os.getenv("ENABLE_LLM_DESCRIPTION", "True").lower() in ("true", "1", "yes"),
    "stream": os.getenv("LLM_STREAM", "False").lower() in ("true", "1", "yes"),
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "250")),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.6")),
    "top_p": float(os.getenv("LLM_TOP_P", "0.9")),
    "repetition_penalty": float(os.getenv("LLM_REPETITION_PENALTY", "1.2")),
}


# Cleanup Settings
CLEANUP_CONFIG = {
    "retention_hours": int(os.getenv("UPLOAD_RETENTION_HOURS", "2")),
    "auto_cleanup": os.getenv("AUTO_CLEANUP_ENABLED", "True").lower() in ("true", "1", "yes"),
}


# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "log_file": str(LOG_DIR / os.getenv("LOG_FILE", "app.log").split("/")[-1]),
}


# Visualization Settings
VISUALIZATION = {
    "font_scale": float(os.getenv("VISUALIZATION_FONT_SCALE", "0.6")),
    "thickness": int(os.getenv("VISUALIZATION_THICKNESS", "2")),
}


# Emotion Colors (BGR format for OpenCV)
EMOTION_COLORS_BGR = {
    "angry": (54, 65, 255),       # #FF4136
    "disgust": (182, 89, 155),    # #9B59B6
    "fear": (61, 61, 61),         # #3D3D3D
    "happy": (64, 204, 46),       # #2ECC40
    "sad": (217, 116, 0),         # #0074D9
    "surprise": (27, 133, 255),   # #FF851B
    "neutral": (170, 170, 170),   # #AAAAAA
    "unknown": (255, 255, 255),   # White
}


# Emotion Colors (HEX format for web)
EMOTION_COLORS_HEX = {
    "angry": "#FF4136",
    "disgust": "#9B59B6",
    "fear": "#3D3D3D",
    "happy": "#2ECC40",
    "sad": "#0074D9",
    "surprise": "#FF851B",
    "neutral": "#AAAAAA",
}


# Emotion list
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# Helper function to validate configuration
def validate_config():
    """Validate GPU availability."""
    errors = []
    
    if MODEL_CONFIG["use_gpu"]:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("⚠️  USE_GPU=True but CUDA is not available.")
        except ImportError:
            errors.append("⚠️  PyTorch not installed but USE_GPU=True")
    
    return errors


# Print configuration summary on import (optional, for debugging)
if __name__ == "__main__":
    print("\n" + "="*60)
    print("IMAGE DESCRIPTION BOT - Configuration Summary")
    print("="*60)
    print(f"\n📁 Directories:")
    print(f"   BASE_DIR:      {BASE_DIR}")
    print(f"   INSTANCE_DIR:  {INSTANCE_DIR}")
    print(f"   UPLOAD_DIR:    {UPLOAD_DIR}")
    print(f"   LOG_DIR:       {LOG_DIR}")
    
    print(f"\n⚙️  Flask Config:")
    print(f"   DEBUG:         {DEBUG}")
    print(f"   MAX_UPLOAD:    {MAX_CONTENT_LENGTH / (1024*1024):.0f} MB")
    
    print(f"\n🤖 Model Config:")
    print(f"   Model:         {MODEL_CONFIG['mistral_model_id']}")
    print(f"   GPU:           {MODEL_CONFIG['use_gpu']}")
    print(f"   4-bit Quant:   {MODEL_CONFIG['use_4bit_quantization']}")
    
    print(f"\n🎨 Face Detection:")
    print(f"   Backend:       {FACE_DETECTION['backend']}")
    print(f"   Margin:        {FACE_DETECTION['margin']:.0%}")
    
    print(f"\n💬 LLM Generation:")
    print(f"   Enabled:       {LLM_CONFIG['enable_description']}")
    print(f"   Max Tokens:    {LLM_CONFIG['max_new_tokens']}")
    print(f"   Temperature:   {LLM_CONFIG['temperature']}")
    
    print(f"\n🧹 Cleanup:")
    print(f"   Auto-cleanup:  {CLEANUP_CONFIG['auto_cleanup']}")
    print(f"   Retention:     {CLEANUP_CONFIG['retention_hours']} hours")
    
    # Validate
    validation_errors = validate_config()
    if validation_errors:
        print(f"\n❌ Configuration Errors:")
        for error in validation_errors:
            print(f"   {error}")
    else:
        print(f"\n✅ Configuration validated successfully!")
    
    print("="*60 + "\n")
