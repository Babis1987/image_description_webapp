"""
Description generator module using Mistral 7B LLM.

Generates natural language descriptions from face detection and analysis results.
"""

import torch
from typing import List, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread

from config import MODEL_CONFIG, LLM_CONFIG


class DescriptionGenerator:
    """
    LLM-based description generator with dual model support.
    
    Supports two models:
    - Mistral 7B Instruct v0.3 (high quality, streaming)
    - FLAN-T5-base (lighter, faster)
    
    Model selection via config.py or constructor parameter.
    Supports runtime model switching with automatic VRAM cleanup.

    """
    
    def __init__(self, model_type: str = None, lazy_load: bool = False):
        """
        Initialize LLM model.
        
        Args:
            model_type: 'mistral' or 'flan-t5'. If None, uses config default.
            lazy_load: If True, delays model loading until first generate call.
                      Useful for faster app startup and testing.
        
        Uses configuration from config.py:
        - MODEL_CONFIG["model_type"] - default model (if model_type=None)
        - MODEL_CONFIG["{model}_model_id"]
        - MODEL_CONFIG["use_gpu"]
        - MODEL_CONFIG["use_4bit_quantization"]
        """
        self.model_type = model_type or MODEL_CONFIG["model_type"]
        self.use_gpu = MODEL_CONFIG["use_gpu"]
        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        
        # Model placeholders
        self.model = None
        self.tokenizer = None
        self.model_id = None
        
        if not lazy_load:
            print(f"🔧 Initializing DescriptionGenerator with {self.model_type}...")
            print(f"🖥️  Device: {self.device}")
            self._load_model()
        else:
            print(f"🔧 DescriptionGenerator initialized (lazy mode - will load {self.model_type} on first use)")
    
    def _load_model(self):
        """Load the appropriate model based on config."""
        if self.model is not None:
            return  # Already loaded
        
        print(f"📥 Loading {self.model_type} model...")
        
        if self.model_type == "mistral":
            self._load_mistral()
        elif self.model_type == "flan-t5":
            self._load_flan_t5()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'mistral' or 'flan-t5'")
    
    def _load_mistral(self):
        """Load Mistral 7B with 4-bit quantization."""
        self.model_id = MODEL_CONFIG["mistral_model_id"]
        
        # Configure 4-bit quantization (from notebook)
        if MODEL_CONFIG["use_4bit_quantization"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            bnb_config = None
        
        print(f"📥 Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto" if self.use_gpu else None,
            trust_remote_code=True
        )
        
        if not self.use_gpu or bnb_config is None:
            self.model = self.model.to(self.device)
        
        print(f"✅ Mistral loaded on {self.device}")
    
    def _load_flan_t5(self):
        """Load FLAN-T5 model (from notebook)."""
        self.model_id = MODEL_CONFIG["flan_t5_model_id"]
        
        print(f"📥 Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if self.use_gpu else torch.float32,
            device_map="auto" if self.use_gpu else None
        )
        
        if not self.use_gpu:
            self.model = self.model.to(self.device)
        
        print(f"✅ FLAN-T5 loaded on {self.device}")
    
    def _unload_model(self):
        """
        Unload current model and free VRAM/RAM.
        
        Important for switching models without running out of memory.
        """
        if self.model is not None:
            print(f"🗑️  Unloading {self.model_type} model...")
            
            # Delete model and tokenizer
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.model_id = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"✅ VRAM cleared")
            else:
                print(f"✅ RAM cleared")
    
    def switch_model(self, model_type: str):
        """
        Switch to a different model at runtime.
        
        Automatically unloads current model and clears VRAM before loading new one.
        
        Args:
            model_type: 'mistral' or 'flan-t5'
        
        Example:
            ```python
            gen = DescriptionGenerator(model_type="flan-t5")
            desc1 = gen.generate_description(faces)  # Uses FLAN-T5
            
            gen.switch_model("mistral")  # Unloads FLAN-T5, loads Mistral
            desc2 = gen.generate_description(faces)  # Uses Mistral
            ```
        """
        if model_type not in ["mistral", "flan-t5"]:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'mistral' or 'flan-t5'")
        
        if model_type == self.model_type and self.model is not None:
            print(f"ℹ️  Already using {model_type} model")
            return
        
        print(f"\n🔄 Switching from {self.model_type} to {model_type}...")
        
        # Unload current model
        self._unload_model()
        
        # Update model type
        self.model_type = model_type
        
        # Load new model
        self._load_model()
        
        print(f"✅ Switched to {model_type}\n")
    
    @staticmethod
    def _summarize_faces(faces: List[Dict[str, Any]]) -> str:
        """
        Convert face detection results to LLM-friendly text summary.
        
        Args:
            faces: List of face detection/analysis results from FaceAnalyzer
        
        Returns:
            Multi-line string summary with format:
            "Face 1: age~28, gender=Woman, dominant_emotion=happy, position=center"
        """
        if not faces:
            return "No faces/objects detected."

        lines = []
        for i, f in enumerate(faces, start=1):
            age = f.get("age")
            gender = f.get("gender")
            dominant = f.get("emotion_label")
            pos = f.get("position", {}).get("position") if isinstance(f.get("position"), dict) else f.get("position")

            extra = []
            if age is not None: 
                extra.append(f"age~{age}")
            if gender is not None: 
                extra.append(f"gender={gender}")
            if dominant is not None: 
                extra.append(f"dominant_emotion={dominant}")
            if pos: 
                extra.append(f"position={pos}")

            lines.append(f"Face {i}: " + ", ".join(extra))

        return "\n".join(lines)
    
    def generate_description(
        self,
        detections: List[Dict[str, Any]],
        stream: bool = None
    ) -> str:
        """
        Generate natural language description from face detections.
        
        Automatically uses the appropriate method based on model_type.
        Handles lazy loading if model not yet loaded.
        
        Args:
            detections: List of face detection/analysis results from FaceAnalyzer
            stream: If True, streams output token-by-token.
                   If None, uses config LLM_CONFIG["stream"]
                   (Mistral only - FLAN-T5 ignores this)
        
        Returns:
            Natural language description string
        """
        # Lazy load if needed
        if self.model is None:
            self._load_model()
        
        if self.model_type == "mistral":
            return self._generate_mistral(detections, stream)
        elif self.model_type == "flan-t5":
            return self._generate_flan_t5(detections)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _generate_flan_t5(
        self,
        detections: List[Dict[str, Any]]
    ) -> str:
        """
        Generate description using FLAN-T5.
        
        FLAN-T5 doesn't support streaming, so stream parameter is ignored.
        """
        summary = self._summarize_faces(detections)
        
        # Few-shot prompt for FLAN-T5 (from notebook - unchanged)
        prompt = f"""Convert face detection data to natural image descriptions.

Data: Face 1: age~28, gender=Woman, dominant_emotion=Happy, position=center
Description: The image shows a woman in her late twenties appears at the center of the frame with a joyful, happy expression.

Data: Face 1: age~45, gender=Man, dominant_emotion=sad, position=left
Description: The image captures a middle-aged man on the left side displaying a sad, melancholic demeanor.

Data: Face 1: age~35, gender=Woman, dominant_emotion=surprise, position=right
Description: The image shows woman in her mid-thirties positioned on the right shows a surprised, wide-eyed expression.

Data: Face 1: age~22, gender=Man, dominant_emotion=angry, position=center
Description: The frame shows a young man around 22 years old at the center with an angry, intense expression.

Data: Face 1: age~50, gender=Woman, dominant_emotion=fear, position=left
Description: The picture shows a woman in her fifties appears on the left side with a fearful, anxious expression.

Data: Face 1: age~30, gender=Man, dominant_emotion=neutral, position=right
Description: The image shows a man approximately 30 years old is positioned on the right displaying a neutral, calm expression.

Data: Face 1: age~40, gender=Woman, dominant_emotion=disgust, position=center
Description: The image shows a woman around 40 years old at the center with a disgusted facial expression.

Data: {summary}
Description:"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        # Use LLM_CONFIG for generation params (but adapt for T5)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=min(LLM_CONFIG["max_new_tokens"], 80),  # T5 works better with shorter outputs
            do_sample=True,
            temperature=LLM_CONFIG["temperature"],
            top_p=LLM_CONFIG["top_p"],
            repetition_penalty=2.0,  # Higher for T5
            no_repeat_ngram_size=3,
        )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return text
    
    def _generate_mistral(
        self,
        detections: List[Dict[str, Any]],
        stream: bool = None
    ) -> str:
        """
        Generate natural language description from face detections.
        
        Args:
            detections: List of face detection/analysis results
            stream: If True, streams output token-by-token.
                   If None, uses config LLM_CONFIG["stream"]
        
        Returns:
            Natural language description string
        """
        use_stream = stream if stream is not None else LLM_CONFIG["stream"]
        summary = self._summarize_faces(detections)
        
        # Mistral uses chat template with system and user messages (from notebook)
        messages = [
            {
                "role": "system",
                "content": "You are an expert at converting structured face detection data into natural, descriptive sentences about images. Create vivid, informative descriptions that capture age, gender, emotional state, and spatial positioning for all faces."
            },
            {
                "role": "user",
                "content": f"""Convert face detection data into natural, structured descriptions. Follow these examples:

Example 1:
Data: Face 1: age~28, gender=Woman, dominant_emotion=happy, position=center
Description: The image captures a joyful young woman, approximately 28 years old, prominently positioned at the center of the frame. Her facial expression radiates happiness, with visible indicators of genuine positive emotion. The central framing suggests she is the primary subject of focus.

Example 2 (Two faces):
Data: Face 1: age~45, gender=Man, dominant_emotion=sad, position=left
Face 2: age~32, gender=Woman, dominant_emotion=neutral, position=right
Description: The composition features two individuals positioned on opposite sides of the frame. On the left, a middle-aged man around 45 years old displays a melancholic expression, his sadness evident in his facial features. Contrasting with this, a woman in her early thirties occupies the right side of the frame, maintaining a neutral, composed demeanor that balances the emotional weight of the scene.

Example 3:
Data: Face 1: age~35, gender=Woman, dominant_emotion=surprise, position=center
Face 2: age~28, gender=Man, dominant_emotion=happy, position=left
Face 3: age~40, gender=Man, dominant_emotion=neutral, position=right
Description: This dynamic group composition captures three distinct individuals with varied emotional states. At the center, a woman in her mid-thirties exhibits a surprised expression, her astonishment forming the focal point of the scene. To her left, a younger man around 28 displays visible happiness, his joyful demeanor adding warmth to the composition. On the right side, a man approximately 40 years old maintains a neutral expression, providing emotional stability to the triangular arrangement of subjects.

Now convert this data:
Data: {summary}
Description:"""
            }
        ]
        
        # Apply chat template (from notebook)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        if use_stream:
            # Use TextIteratorStreamer for real streaming (from notebook)
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            # Run generation in separate thread
            generation_kwargs = dict(
                inputs,
                max_new_tokens=LLM_CONFIG["max_new_tokens"],
                do_sample=True,
                temperature=LLM_CONFIG["temperature"],
                top_p=LLM_CONFIG["top_p"],
                repetition_penalty=LLM_CONFIG["repetition_penalty"],
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream output
            generated_text = ""
            for text in streamer:
                generated_text += text
            
            thread.join()
            return generated_text
        else:
            # Non-streaming version (from notebook)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_CONFIG["max_new_tokens"],
                do_sample=True,
                temperature=LLM_CONFIG["temperature"],
                top_p=LLM_CONFIG["top_p"],
                repetition_penalty=LLM_CONFIG["repetition_penalty"],
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
