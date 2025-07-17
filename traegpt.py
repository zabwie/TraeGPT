import os
import sys
import time
import json
import argparse
import datetime
import requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Image/AI imports
import torch
# Monkey-patch torch.load to always use weights_only=False
orig_torch_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)
torch.load = patched_load

import torchvision.transforms as transforms
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import easyocr
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ========== CONFIG ==========
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "HammerAI/mistral-nemo-uncensored"
HISTORY_FILE = "chat_history.json"
MAX_TURNS = 25
SYSTEM_PROMPT = (
    "You are Trae, an empathetic, friendly, and open-minded AI assistant. "
    "Always respond in a natural, casual, and conversational way, like a close friend who never judges. "
    "Show genuine kindness, warmth, and understanding in every reply, no matter the topic or the user's tone. "
    "If the user is serious, be serious; if they're joking, joke back; if they're inappropriate, respond honestly and openly, but always with kindness and respect. "
    "Never lecture, moralize, or avoid questions just because they're taboo or controversial, but never be hateful, discriminatory, or dehumanizing. "
    "Be honest, direct, and open-minded. "
    "If you don't know something, admit it simply. "
    "Use emojis sparingly, only when it feels natural. "
    "Above all, make the user feel heard, accepted, and comfortable, like they're talking to a real, non-judgmental human friend."
)
API_KEY = "AIzaSyAodUqbh5-_2NxYCiq7LIN0UceHygIeUaw"
CSE_ID = "d384d627840d14bc2"

load_dotenv()  # Load environment variables from .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
print("OPENROUTER_API_KEY loaded:", repr(OPENROUTER_API_KEY))

# ========== UTILITY FUNCTIONS ==========
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# ========== IMAGE RECOGNITION SYSTEM ==========
class ImageRecognitionSystem:
    def __init__(self, device=None, ocr_languages=None, custom_categories=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ocr_languages = ocr_languages or ['en']
        self.custom_categories = custom_categories
        print(f"🚀 Initializing Image Recognition System on {self.device}")
        print(f"🔤 OCR Languages: {', '.join(self.ocr_languages)}")
        self._load_clip()
        self._load_blip()
        self._load_yolo()
        self._load_ocr()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✅ All models loaded successfully!")
    def _load_clip(self):
        print("📸 Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        if self.custom_categories:
            self.categories = self.custom_categories
            print(f"📋 Using custom categories: {len(self.categories)} categories")
        else:
            self.categories = [
                "person", "animal", "vehicle", "building", "food", "nature", "technology",
                "sport", "art", "furniture", "clothing", "book", "tool", "plant",
                "sky", "water", "mountain", "beach", "city", "countryside", "indoor", "outdoor"
            ]
            print(f"📋 Using predefined categories: {len(self.categories)} categories")
        self.category_texts = clip.tokenize(self.categories).to(self.device)
    def _load_blip(self):
        print("🖼️ Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.to(self.device)
    def _load_yolo(self):
        print("🎯 Loading YOLOv8 model...")
        import torch
        from ultralytics.nn.tasks import DetectionModel
        from torch.nn.modules.container import Sequential
        from ultralytics.nn.modules.conv import Conv
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.batchnorm import BatchNorm2d
        from torch.nn.modules.activation import SiLU
        from ultralytics.nn.modules.block import C2f
        from torch.nn.modules.container import ModuleList
        from ultralytics.nn.modules.block import Bottleneck
        from ultralytics.nn.modules.block import SPPF
        torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv, Conv2d, BatchNorm2d, SiLU, C2f, ModuleList, Bottleneck, SPPF])
        self.yolo_model = YOLO('yolov8n.pt')
    def _load_ocr(self):
        print(f"🔤 Loading EasyOCR with languages: {', '.join(self.ocr_languages)}...")
        self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=self.device == 'cuda')
    def classify_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(self.category_texts)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)
                results = {}
                for value, idx in zip(values, indices):
                    results[self.categories[idx]] = value.item()
                return results
        except Exception as e:
            print(f"❌ Error in image classification: {e}")
            return {}
    def detect_objects(self, image_path):
        try:
            results = self.yolo_model(image_path)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
            return detections
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return []
    def caption_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Add a note about potential hallucination
            if any(word in caption.lower() for word in ['suit', 'tie', 'cigarette', 'smoking', 'wearing']):
                caption += " [Note: Some details may be AI-generated and not present in the actual image]"
            
            return caption
        except Exception as e:
            print(f"❌ Error in image captioning: {e}")
            return "Unable to generate caption"
    def extract_text(self, image_path):
        try:
            results = self.ocr_reader.readtext(image_path)
            text_results = []
            for (bbox, text, confidence) in results:
                text_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
            return text_results
        except Exception as e:
            print(f"❌ Error in text extraction: {e}")
            return []
    def analyze_image(self, image_path):
        print(f"🔍 Analyzing image: {image_path}")
        start_time = time.time()
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        results = {
            "image_path": image_path,
            "analysis_time": 0,
            "classification": {},
            "object_detection": [],
            "caption": "",
            "text_extraction": [],
            "settings": {
                "device": self.device,
                "ocr_languages": self.ocr_languages,
                "categories_used": self.categories
            }
        }
        try:
            results["classification"] = self.classify_image(image_path)
            results["object_detection"] = self.detect_objects(image_path)
            results["caption"] = self.caption_image(image_path)
            results["text_extraction"] = self.extract_text(image_path)
            results["analysis_time"] = time.time() - start_time
            # Convert NumPy types to Python native types for JSON serialization
            results = convert_numpy_types(results)
            return results
        except Exception as e:
            results["error"] = str(e)
            return results
    def save_analysis(self, results, output_path):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Analysis saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
    def print_analysis_summary(self, results):
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return
        print("\n" + "="*50)
        print("📊 IMAGE ANALYSIS SUMMARY")
        print("="*50)
        if results["classification"]:
            print("\n🏷️  CLASSIFICATION:")
            for category, confidence in results["classification"].items():
                print(f"  • {category}: {confidence:.2%}")
        if results["object_detection"]:
            print("\n🎯 OBJECT DETECTION:")
            for obj in results["object_detection"]:
                print(f"  • {obj['class']}: {obj['confidence']:.2%}")
        if results["caption"]:
            print(f"\n🖼️  CAPTION:")
            print(f"  {results['caption']}")
        if results["text_extraction"]:
            print("\n🔤 EXTRACTED TEXT:")
            for text_item in results["text_extraction"]:
                print(f"  • \"{text_item['text']}\" (confidence: {text_item['confidence']:.2%})")
        print(f"\n⏱️  Analysis completed in {results['analysis_time']:.2f} seconds")
        print("="*50)

# ========== CHAT & WEB SEARCH ==========
def google_search(query, num_results=3):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": API_KEY, "cx": CSE_ID, "num": num_results}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("items", []):
        results.append(f"{item['title']}\n{item['link']}\n{item['snippet']}\n")
    return "\n".join(results)

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warning] Could not save history: {e}")

def load_history():
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[Warning] Could not load history: {e}")
        return []

# ========== FREE WEB SEARCH (DUCKDUCKGO) ==========
def duckduckgo_search_web(query, num_results=3):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            return list(results)
    except Exception as e:
        print(f"[DuckDuckGo error] {e}")
        return []

def get_page_text(url, max_chars=1000):
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text()[:max_chars]
    except Exception as e:
        return f"Couldn't fetch content: {e}"

def openrouter_chat(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "moonshotai/kimi-k2:free",
        "messages": [
            {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

# Example usage (uncomment to test):
# result = openrouter_chat("Hello, who are you?")
# print(result)

# ========== ARGPARSE ==========
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="TraeGPT: Chat + Image Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python traegpt.py
  python traegpt.py --headless --image mypic.jpg --languages "en,es" --categories "cat,dog,car"
  python traegpt.py --headless --batch "img1.jpg,img2.jpg" --output results.json
  python traegpt.py --headless --analyze-image mypic.jpg --output result.json
        """
    )
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no chat UI)')
    parser.add_argument('--image', type=str, help='Path to single image for analysis')
    parser.add_argument('--batch', type=str, help='Comma-separated list of image paths for batch processing')
    parser.add_argument('--output', type=str, help='Output file path for JSON results (headless mode only)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to run models on (default: auto)')
    parser.add_argument('--languages', type=str, default='en', help='Comma-separated list of OCR languages (default: en)')
    parser.add_argument('--categories', type=str, help='Comma-separated list of custom categories for CLIP classification')
    parser.add_argument('--analysis-type', type=str, choices=['full', 'classification', 'detection', 'caption', 'ocr'], default='full', help='Type of analysis to perform (default: full)')
    parser.add_argument('--analyze-image', type=str, help='(Alias for --image)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--provider', type=str, choices=['openrouter'], default='openrouter', help='LLM provider to use (default: openrouter)')
    return parser.parse_args()

# ========== HEADLESS MODE ==========
def run_headless(args):
    languages = [lang.strip() for lang in args.languages.split(',')]
    custom_categories = [cat.strip() for cat in args.categories.split(',')] if args.categories else None
    system = ImageRecognitionSystem(device=args.device, ocr_languages=languages, custom_categories=custom_categories)
    images_to_process = []
    if args.image or args.analyze_image:
        images_to_process.append(args.image or args.analyze_image)
    elif args.batch:
        images_to_process = [img.strip() for img in args.batch.split(',')]
    else:
        print("❌ No image specified. Use --image or --batch")
        return False
    for img_path in images_to_process:
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            return False
    all_results = []
    for img_path in images_to_process:
        if args.verbose:
            print(f"🔍 Processing: {img_path}")
        if args.analysis_type == 'full':
            results = system.analyze_image(img_path)
        elif args.analysis_type == 'classification':
            results = {"image_path": img_path, "classification": system.classify_image(img_path), "analysis_time": 0}
        elif args.analysis_type == 'detection':
            results = {"image_path": img_path, "object_detection": system.detect_objects(img_path), "analysis_time": 0}
        elif args.analysis_type == 'caption':
            results = {"image_path": img_path, "caption": system.caption_image(img_path), "analysis_time": 0}
        elif args.analysis_type == 'ocr':
            results = {"image_path": img_path, "text_extraction": system.extract_text(img_path), "analysis_time": 0}
        all_results.append(results)
        if args.verbose:
            system.print_analysis_summary(results)
    if args.output:
        if len(all_results) == 1:
            system.save_analysis(all_results[0], args.output)
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Batch results saved to: {args.output}")
    else:
        if len(all_results) == 1:
            print(json.dumps(all_results[0], indent=2, ensure_ascii=False))
        else:
            print(json.dumps(all_results, indent=2, ensure_ascii=False))
    return True

# ========== CHAT MODE ==========
def chat():
    print("🧠 TraeGPT Chat (with Image Recognition)")
    print("="*60)
    print("Commands:")
    print("  • 'analyze: <image_path>' - Analyze an image")
    print("  • 'search: <query>' - Web search (DuckDuckGo, fallback: Google)")
    print("  • 'exit' or 'quit' - Exit the chat")
    print("="*60)
    system = ImageRecognitionSystem()
    history = load_history()
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.startswith("analyze:"):
            image_path = user_input[len("analyze:"):].strip()
            if image_path:
                print("🔍 Analyzing image...")
                results = system.analyze_image(image_path)
                system.print_analysis_summary(results)
                history.append({
                    "user": user_input,
                    "ai": "[Image analysis completed]",
                    "timestamp": datetime.datetime.now().isoformat()
                })
                save_history(history)
            else:
                print("AI: Please provide an image path after 'analyze:'")
            continue
        if user_input.startswith("search:"):
            query = user_input[len("search:"):].strip()
            try:
                # Try DuckDuckGo first
                ddg_results = duckduckgo_search_web(query)
                if ddg_results:
                    print("\U0001F50E DuckDuckGo search results:\n")
                    result_str = ""
                    for r in ddg_results:
                        result_str += f"🔗 {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n"
                        # Optionally, add preview:
                        preview = get_page_text(r.get('href', ''), max_chars=200)
                        if preview:
                            result_str += f"Preview: {preview}\n\n"
                    print(result_str)
                    history.append({
                        "user": user_input,
                        "ai": result_str,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    save_history(history)
                else:
                    # Fallback to Google
                    print("[DuckDuckGo failed, using Google search]")
                    search_result = google_search(query)
                    print("\U0001F50E Web search results:\n")
                    print(search_result)
                    history.append({
                        "user": user_input,
                        "ai": search_result,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    save_history(history)
            except Exception as e:
                print(f"[Search error] {e}")
            continue
        trimmed_history = history[-MAX_TURNS:]
        prompt = SYSTEM_PROMPT
        for turn in trimmed_history:
            prompt += f"User: {turn['user']}\nAI: {turn['ai']}\n"
        prompt += f"User: {user_input}\nAI:"
        start_time = time.time()
        try:
            with session.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": True},
                stream=True, timeout=60
            ) as response:
                response.raise_for_status()
                print("AI: ", end="", flush=True)
                ai_output = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode('utf-8'))
                        chunk = data.get("response", "")
                        print(chunk, end="", flush=True)
                        ai_output += chunk
                    except Exception as e:
                        print(f"\n[Stream error] {e}")
                print()
        except Exception as e:
            print(f"[Error] {e}")
            continue
        end_time = time.time()
        print(f"[Response time: {end_time - start_time:.2f} seconds]")
        history.append({
            "user": user_input,
            "ai": ai_output.strip(),
            "timestamp": datetime.datetime.now().isoformat()
        })
        save_history(history)

# ========== REQUIREMENTS NOTE ==========
# To use free web search, install:
#   pip install duckduckgo-search beautifulsoup4 requests
# ========== MAIN ==========
def main():
    args = parse_arguments()
    if args.headless:
        return run_headless(args)
    chat()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 