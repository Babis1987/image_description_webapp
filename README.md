# Image Description Web App

A Flask-based web application that performs **face detection**, **emotion recognition**, and generates **natural language descriptions** of detected faces using Large Language Models (LLMs).

---

## 📋 Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Models Used](#models-used)
- [Project Structure](#project-structure)
- [License](#license)

---

## ✨ Features

- **Face Detection**: Detects faces in uploaded images using MediaPipe
- **Emotion Recognition**: Analyzes emotions, age, and gender using DeepFace
- **Natural Language Descriptions**: Generates human-readable descriptions using LLMs (Mistral 7B or FLAN-T5)
- **Multi-Model Support**: Switch between Mistral 7B and FLAN-T5 models
- **Session Management**: Keep history of analyzed images across page refreshes
- **Visual Annotations**: Displays bounding boxes with emotion labels on detected faces
- **Responsive UI**: Modern interface with background customization and mobile support

---

## 🛠️ Technology Stack

### Backend
- **Python 3.10.19**
- **Flask 3.1.2** - Web framework
- **DeepFace 0.0.79** - Face detection and emotion analysis
- **Transformers 4.57.3** - LLM inference
- **PyTorch 2.7.1** - Deep learning framework
- **OpenCV 4.8.1** - Image processing

### Frontend
- HTML5, CSS3, Jinja2 templates
- Responsive flexbox/grid layout

### AI Models
- **MediaPipe** - Face detection backend
- **DeepFace** - Emotion, age, gender recognition
- **Mistral 7B Instruct v0.3** - High-quality description generation
- **FLAN-T5-base** - Lightweight alternative for faster inference

---

## 📦 Prerequisites

- **Python**: Version 3.10.x (tested with 3.10.19)
- **CUDA** (optional): For GPU acceleration with PyTorch
- **RAM**: Minimum 8GB (16GB recommended for Mistral 7B)
- **Storage**: ~10GB for models and dependencies

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd image_description_webapp
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installation may take 10-15 minutes due to large dependencies (PyTorch, Transformers, TensorFlow).

### 4. Download Models

Models will be automatically downloaded on first run:
- **Mistral 7B**: ~14GB (downloaded from Hugging Face)
- **FLAN-T5-base**: ~1GB
- **DeepFace models**: Downloaded automatically by DeepFace library

---

## ⚙️ Configuration

All configuration is managed through `config.py`. Key settings:

### Environment Variables (Optional)

Create a `.env` file in the project root:

```env
# Flask Settings
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True

# Upload Settings
MAX_CONTENT_LENGTH_MB=16
UPLOAD_DIR=uploads

# Face Detection
FACE_DETECTION_BACKEND=mediapipe
FACE_DETECTION_MARGIN=0.20

# Face Analysis
FACE_ANALYSIS_ACTIONS=emotion,age,gender

# LLM Model
MODEL_TYPE=mistral
USE_GPU=True
USE_4BIT_QUANTIZATION=True
```

### Default Configuration

If `.env` is not provided, the app uses sensible defaults from `config.py`:
- **Face Detection Backend**: MediaPipe
- **Face Analysis Actions**: Emotion, Age, Gender
- **Default LLM Model**: Mistral 7B
- **GPU Usage**: Enabled (if available)
- **4-bit Quantization**: Enabled (for Mistral, reduces VRAM)

---

## 🎯 Usage

### 1. Start the Application

```bash
python app.py
```

The app will start on `http://127.0.0.1:5000`

### 2. Upload Images

1. Navigate to the chat interface (`/chat`)
2. Select a model (Mistral or FLAN-T5)
3. Click **"Choose Files"** and select one or more images
4. Click **"Analyze"** to process

### 3. View Results

- **Left Sidebar**: Shows uploaded filenames
- **Right Panel**: Displays annotated images with emotion labels and generated descriptions
- **History**: All processed images are saved in the session
- **Clear History**: Remove all results and start fresh

### 4. Endpoints

- `/` - Landing page
- `/chat` - Main interface for image upload and analysis
- `/instructions` - Usage instructions
- `/clear_history` - Clear session history

---

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────────┐
│   Flask App     │
│    (app.py)     │
└────────┬────────┘
         │
         ├──────────────────────────────────┐
         │                                  │
    ┌────▼────┐                      ┌─────▼─────┐
    │ Pipeline│                      │  Session  │
    │ (core/) │                      │Management │
    └────┬────┘                      └───────────┘
         │
    ┌────┴───────────────────────┐
    │                            │
┌───▼───────┐            ┌───────▼────────┐
│   Face    │            │  Description   │
│ Detection │───────────▶│   Generator    │
│(MediaPipe)│            │ (Mistral/T5)   │
└───┬───────┘            └────────────────┘
    │
┌───▼───────┐
│   Face    │
│ Analysis  │
│(DeepFace) │
└───────────┘
```

### Core Modules

#### 1. **Face Detection** (`core/face_detection.py`)
- Uses **MediaPipe** backend (configurable)
- Detects face bounding boxes with configurable margin
- Returns face coordinates and cropped face images

#### 2. **Face Analysis** (`core/face_analysis.py`)
- Uses **DeepFace** for emotion recognition
- Analyzes: Emotion (7 classes), Age (range), Gender
- Emotion model: Trained on FER2013 dataset

#### 3. **Description Generator** (`core/description_generator.py`)
- **Mistral 7B Instruct v0.3**: High-quality, context-aware descriptions
  - 4-bit quantization for memory efficiency
  - Supports streaming generation
- **FLAN-T5-base**: Faster, lighter alternative
  - Good for resource-constrained environments

#### 4. **Pipeline** (`core/pipeline.py`)
- Orchestrates the entire workflow:
  1. Detect faces in image
  2. Analyze each face (emotion, age, gender)
  3. Generate natural language description
  4. Annotate image with results

#### 5. **Visualization** (`core/visualization.py`)
- Draws bounding boxes around detected faces
- Adds emotion labels with confidence scores
- Creates annotated output images

---

## 🤖 Models Used

### Face Detection
- **Backend**: MediaPipe Face Detection
  - Fast and accurate
  - Optimized for real-time performance
  - Works on CPU efficiently

### Emotion Recognition
- **Framework**: DeepFace
- **Emotion Model**: VGG-Face trained on FER2013
  - 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
  - Age estimation: Regression model
  - Gender classification: Binary classifier

### Language Models
1. **Mistral 7B Instruct v0.3**
   - Parameters: 7 billion
   - Context window: 8k tokens
   - Quantization: 4-bit (reduces to ~4GB VRAM)
   - Best for: High-quality, context-aware descriptions

2. **FLAN-T5-base**
   - Parameters: 250 million
   - Faster inference (~2-3x speed)
   - Lower memory footprint (~1GB)
   - Best for: Quick testing, resource-constrained setups

---

## 📁 Project Structure

```
image_description_webapp/
├── app.py                      # Flask application entry point
├── config.py                   # Configuration and settings
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (optional)
├── .gitignore                  # Git ignore patterns
│
├── core/                       # Core processing modules
│   ├── face_detection.py       # Face detection logic
│   ├── face_analysis.py        # Emotion/age/gender analysis
│   ├── description_generator.py # LLM text generation
│   ├── pipeline.py             # Main processing pipeline
│   └── visualization.py        # Image annotation
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Landing page
│   ├── chat.html              # Main interface
│   └── instructions.html      # Help page
│
├── static/                    # Static assets
│   ├── css/
│   │   ├── chat.css           # Chat interface styles
│   │   └── index.css          # Landing page styles
│   └── images/
│       └── chat_background.png # Background image
│
└── instance/                  # Runtime data (gitignored)
    ├── uploads/
    │   ├── original/          # Original uploaded images
    │   └── processed/         # Annotated output images
    └── logs/                  # Application logs
```

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Solution: Enable 4-bit quantization in config.py
- Or: Switch to FLAN-T5 model (less VRAM)

**2. Model Download Fails**
- Check internet connection
- Models download to `~/.cache/huggingface/`
- Ensure ~15GB free disk space

**3. Face Detection Not Working**
- Verify image format (JPEG, PNG supported)
- Check image resolution (very low resolution may fail)
- Try different detection backend in config.py

**4. Slow Inference**
- Use GPU if available (set `USE_GPU=True`)
- Switch to FLAN-T5 for faster processing
- Reduce image resolution before upload

---

## 📝 License

This project is developed for academic purposes as part of the MSc AI program at University of Essex.

---

## 👥 Contributors

Developed as part of MSc AI coursework - University of Essex

---

## 🙏 Acknowledgments

- **DeepFace**: Serengil, S. I., & Ozpinar, A. (2020). LightFace: A Hybrid Deep Face Recognition Framework. IEEE Xplore.
- **Mistral AI**: For the Mistral 7B Instruct model
- **Hugging Face**: For Transformers library and model hosting
- **MediaPipe**: Google's cross-platform ML solutions

---

**Happy Analyzing! 🎉**
