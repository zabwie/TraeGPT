# TraeGPT - Enhanced AI Assistant with Image Recognition

A powerful AI assistant that combines natural language processing with advanced image recognition capabilities.

## 🚀 Features

### Core Chat Features
- **Natural Language Processing**: Powered by Ollama with HammerAI/mistral-nemo-uncensored model
- **Web Search Integration**: Google Custom Search API for real-time information
- **Chat History**: Persistent conversation history with JSON storage
- **Streaming Responses**: Real-time AI responses with progress indicators

### Advanced Image Recognition
- **🏷️ Image Classification**: CLIP model for semantic image understanding
- **🎯 Object Detection**: YOLOv8 for precise object localization
- **🖼️ Image Captioning**: BLIP model for natural language descriptions
- **🔤 Text Recognition**: EasyOCR for extracting text from images
- **💾 Results Export**: JSON format for analysis results

### CLI Features
- **Headless Mode**: Run without interactive prompts
- **Multi-language OCR**: Support for multiple languages
- **Custom Categories**: Define custom classification categories
- **Batch Processing**: Process multiple images at once
- **Output Control**: Save results to files or print to stdout

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Ollama server running locally

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Manual Installation (if needed)

```bash
# Core dependencies
pip install torch torchvision ftfy regex tqdm pillow numpy opencv-python

# CLIP for image classification
pip install git+https://github.com/openai/CLIP.git

# Transformers and BLIP for image captioning
pip install transformers accelerate
pip install git+https://github.com/salesforce/BLIP.git

# YOLOv8 for object detection
pip install ultralytics

# EasyOCR for text recognition
pip install easyocr

# Additional utilities
pip install requests
```

## 🎯 Usage

### 1. Interactive Mode

```bash
# Original chat interface
python ollama.py

# Enhanced chat with image recognition
python ollama_enhanced.py

# Standalone image recognition system
python image_recognition.py

# Demo the system capabilities
python demo_image_recognition.py
```

### 2. Headless Mode

#### Image Recognition System

```bash
# Basic headless analysis
python image_recognition.py --headless --image path/to/image.jpg

# With custom categories
python image_recognition.py --headless --image path/to/image.jpg --categories "person,dog,car"

# With multiple languages
python image_recognition.py --headless --image path/to/image.jpg --languages "en,es,fr"

# With output file
python image_recognition.py --headless --image path/to/image.jpg --output results.json

# Batch processing
python image_recognition.py --headless --batch "image1.jpg,image2.jpg,image3.jpg"

# Specific analysis type
python image_recognition.py --headless --image path/to/image.jpg --analysis-type classification

# Verbose output
python image_recognition.py --headless --image path/to/image.jpg --verbose
```

#### Enhanced Chat Interface

```bash
# Headless image analysis
python ollama_enhanced.py --headless --analyze-image path/to/image.jpg

# With custom settings
python ollama_enhanced.py --headless --analyze-image path/to/image.jpg --languages "en,es" --categories "person,dog,car"

# With output file
python ollama_enhanced.py --headless --analyze-image path/to/image.jpg --output results.json
```

#### Demo Script

```bash
# Headless demo
python demo_image_recognition.py --headless

# With specific image
python demo_image_recognition.py --headless --image path/to/image.jpg

# With custom settings
python demo_image_recognition.py --headless --languages "en,es" --categories "person,dog,car"
```

### 3. Chat Commands

In the enhanced chat interface, you can use these commands:

- `analyze: path/to/image.jpg` - Analyze an image with all recognition models
- `search: your search query` - Perform web search
- `exit` or `quit` - Exit the chat

### 4. CLI Arguments Reference

#### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--headless` | Run without interactive prompts | False |
| `--device` | Device to run models on (cpu/cuda) | Auto-detect |
| `--languages` | Comma-separated OCR languages | en |
| `--categories` | Comma-separated custom categories | Predefined |
| `--verbose` | Enable verbose output | False |

#### Image Recognition System

| Argument | Description |
|----------|-------------|
| `--image` | Path to single image |
| `--batch` | Comma-separated image paths |
| `--output` | Output file path |
| `--analysis-type` | Type of analysis (full/classification/detection/caption/ocr) |

#### Enhanced Chat Interface

| Argument | Description |
|----------|-------------|
| `--analyze-image` | Path to image for analysis |
| `--output` | Output file path |

### 5. Supported Languages for OCR

EasyOCR supports many languages. Common ones include:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

### 6. Custom Categories Examples

```bash
# Basic categories
--categories "person,dog,car,house"

# Detailed categories
--categories "human,animal,vehicle,building,food,nature,technology"

# Specific use case
--categories "defect,normal,damaged,clean,dirty"
```

## 🔧 Configuration

### API Keys
Update the following in `ollama.py` or `ollama_enhanced.py`:

```python
API_KEY = "your_google_api_key"
CSE_ID = "your_custom_search_engine_id"
```

### Model Configuration
- **Ollama Model**: Change `MODEL` variable to use different models
- **YOLO Model**: Modify `yolo_model = YOLO('yolov8n.pt')` for different sizes
- **CLIP Model**: Change `clip.load("ViT-B/32", device=self.device)` for different variants

## 📊 Output Formats

### Image Analysis Results
```json
{
  "image_path": "path/to/image.jpg",
  "analysis_time": 2.34,
  "classification": {
    "person": 0.85,
    "indoor": 0.72,
    "technology": 0.45
  },
  "object_detection": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": [100, 150, 300, 450]
    }
  ],
  "caption": "A person sitting at a desk with a computer",
  "text_extraction": [
    {
      "text": "Hello World",
      "confidence": 0.95,
      "bbox": [[100, 100], [200, 100], [200, 120], [100, 120]]
    }
  ],
  "settings": {
    "device": "cuda",
    "ocr_languages": ["en", "es"],
    "categories_used": ["person", "dog", "car"]
  }
}
```

## 🎯 Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## ⚡ Performance Tips

1. **GPU Acceleration**: Use CUDA-compatible GPU for faster processing
2. **Model Selection**: Use smaller models (e.g., YOLOv8n) for faster inference
3. **Batch Processing**: Process multiple images in sequence
4. **Memory Management**: Close unused models to free memory
5. **Headless Mode**: Use for automation and scripting

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python image_recognition.py --device cpu
   ```

2. **Model Download Issues**
   ```bash
   # Clear cache and retry
   pip cache purge
   pip install --no-cache-dir -r requirements.txt
   ```

3. **Ollama Connection Error**
   ```bash
   # Start Ollama server
   ollama serve
   ```

4. **Language Support Issues**
   ```bash
   # Use only English for faster processing
   python image_recognition.py --languages en
   ```

### Error Messages

- `Image file not found`: Check file path and permissions
- `Analysis failed`: Check image format and file integrity
- `Model loading error`: Verify internet connection and dependencies
- `No image specified`: Use `--image` or `--batch` in headless mode

## 📈 Performance Benchmarks

Typical processing times (CPU):
- **Classification**: ~0.5s
- **Object Detection**: ~1.2s
- **Captioning**: ~2.0s
- **Text Recognition**: ~1.5s
- **Full Analysis**: ~5.0s

With GPU acceleration, times are reduced by 50-80%.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP**: For image classification capabilities
- **Ultralytics**: For YOLOv8 object detection
- **Salesforce BLIP**: For image captioning
- **JaidedAI EasyOCR**: For text recognition
- **Ollama**: For local LLM inference

---

**Note**: This system requires significant computational resources. For optimal performance, use a machine with at least 8GB RAM and a CUDA-compatible GPU.