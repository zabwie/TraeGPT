import torch
import torchvision.transforms as transforms
import clip
import ftfy
import regex
import tqdm
from PIL import Image
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import easyocr
import os
import json
from typing import List, Dict, Tuple, Optional
import time
import argparse

class ImageRecognitionSystem:
    """
    A comprehensive image recognition system that combines:
    - CLIP for image classification
    - YOLOv8 for object detection
    - BLIP for image captioning
    - EasyOCR for text recognition
    """
    
    def __init__(self, device: str = None, ocr_languages: List[str] = None, custom_categories: List[str] = None):
        """
        Initialize the image recognition system with all models.
        
        Args:
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            ocr_languages: List of language codes for OCR (default: ['en'])
            custom_categories: Custom categories for CLIP classification (default: predefined categories)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ocr_languages = ocr_languages or ['en']
        self.custom_categories = custom_categories
        
        print(f"🚀 Initializing Image Recognition System on {self.device}")
        print(f"🔤 OCR Languages: {', '.join(self.ocr_languages)}")
        
        # Initialize models
        self._load_clip()
        self._load_blip()
        self._load_yolo()
        self._load_ocr()
        
        # Common image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ All models loaded successfully!")
    
    def _load_clip(self):
        """Load CLIP model for image classification."""
        print("📸 Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Use custom categories if provided, otherwise use predefined ones
        if self.custom_categories:
            self.categories = self.custom_categories
            print(f"📋 Using custom categories: {len(self.categories)} categories")
        else:
            # Predefined categories for classification
            self.categories = [
                "person", "animal", "vehicle", "building", "food", "nature", "technology",
                "sport", "art", "furniture", "clothing", "book", "tool", "plant",
                "sky", "water", "mountain", "beach", "city", "countryside", "indoor", "outdoor"
            ]
            print(f"📋 Using predefined categories: {len(self.categories)} categories")
        
        self.category_texts = clip.tokenize(self.categories).to(self.device)
    
    def _load_blip(self):
        """Load BLIP model for image captioning."""
        print("🖼️ Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.to(self.device)
    
    def _load_yolo(self):
        """Load YOLOv8 model for object detection."""
        print("🎯 Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Using nano model for speed
    
    def _load_ocr(self):
        """Load EasyOCR for text recognition."""
        print(f"🔤 Loading EasyOCR with languages: {', '.join(self.ocr_languages)}...")
        self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=self.device == 'cuda')
    
    def classify_image(self, image_path: str) -> Dict[str, float]:
        """
        Classify image using CLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with category names and confidence scores
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(self.category_texts)
                
                # Calculate similarities
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get top 5 results
                values, indices = similarity[0].topk(5)
                
                results = {}
                for value, idx in zip(values, indices):
                    results[self.categories[idx]] = value.item()
                
                return results
                
        except Exception as e:
            print(f"❌ Error in image classification: {e}")
            return {}
    
    def detect_objects(self, image_path: str) -> List[Dict]:
        """
        Detect objects in image using YOLOv8.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected objects with bounding boxes and confidence scores
        """
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
    
    def caption_image(self, image_path: str) -> str:
        """
        Generate caption for image using BLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated caption text
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            print(f"❌ Error in image captioning: {e}")
            return "Unable to generate caption"
    
    def extract_text(self, image_path: str) -> List[Dict]:
        """
        Extract text from image using EasyOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected text with bounding boxes and confidence scores
        """
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
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Perform comprehensive image analysis using all models.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Complete analysis results
        """
        print(f"🔍 Analyzing image: {image_path}")
        start_time = time.time()
        
        # Check if file exists
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
            # Run all analyses
            results["classification"] = self.classify_image(image_path)
            results["object_detection"] = self.detect_objects(image_path)
            results["caption"] = self.caption_image(image_path)
            results["text_extraction"] = self.extract_text(image_path)
            
            results["analysis_time"] = time.time() - start_time
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            return results
    
    def save_analysis(self, results: Dict, output_path: str):
        """
        Save analysis results to JSON file.
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Analysis saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
    
    def print_analysis_summary(self, results: Dict):
        """
        Print a formatted summary of the analysis results.
        
        Args:
            results: Analysis results dictionary
        """
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return
        
        print("\n" + "="*50)
        print("📊 IMAGE ANALYSIS SUMMARY")
        print("="*50)
        
        # Classification results
        if results["classification"]:
            print("\n🏷️  CLASSIFICATION:")
            for category, confidence in results["classification"].items():
                print(f"  • {category}: {confidence:.2%}")
        
        # Object detection results
        if results["object_detection"]:
            print("\n🎯 OBJECT DETECTION:")
            for obj in results["object_detection"]:
                print(f"  • {obj['class']}: {obj['confidence']:.2%}")
        
        # Caption
        if results["caption"]:
            print(f"\n🖼️  CAPTION:")
            print(f"  {results['caption']}")
        
        # Text extraction
        if results["text_extraction"]:
            print("\n🔤 EXTRACTED TEXT:")
            for text_item in results["text_extraction"]:
                print(f"  • \"{text_item['text']}\" (confidence: {text_item['confidence']:.2%})")
        
        print(f"\n⏱️  Analysis completed in {results['analysis_time']:.2f} seconds")
        print("="*50)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TraeGPT Image Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python image_recognition.py

  # Headless mode with single image
  python image_recognition.py --headless --image path/to/image.jpg

  # Headless mode with custom categories
  python image_recognition.py --headless --image path/to/image.jpg --categories "person,dog,car"

  # Headless mode with multiple languages
  python image_recognition.py --headless --image path/to/image.jpg --languages "en,es,fr"

  # Headless mode with output file
  python image_recognition.py --headless --image path/to/image.jpg --output results.json

  # Batch processing
  python image_recognition.py --headless --batch "image1.jpg,image2.jpg,image3.jpg"
        """
    )
    
    # Mode arguments
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode without interactive prompts')
    
    # Input arguments
    parser.add_argument('--image', type=str, 
                       help='Path to single image for analysis')
    parser.add_argument('--batch', type=str, 
                       help='Comma-separated list of image paths for batch processing')
    
    # Output arguments
    parser.add_argument('--output', type=str, 
                       help='Output file path for JSON results (headless mode only)')
    
    # Configuration arguments
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                       help='Device to run models on (default: auto-detect)')
    parser.add_argument('--languages', type=str, default='en',
                       help='Comma-separated list of OCR languages (default: en)')
    parser.add_argument('--categories', type=str,
                       help='Comma-separated list of custom categories for CLIP classification')
    
    # Analysis type arguments
    parser.add_argument('--analysis-type', type=str, 
                       choices=['full', 'classification', 'detection', 'caption', 'ocr'],
                       default='full', help='Type of analysis to perform (default: full)')
    
    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def run_headless_analysis(args):
    """Run analysis in headless mode."""
    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(',')]
    
    # Parse custom categories if provided
    custom_categories = None
    if args.categories:
        custom_categories = [cat.strip() for cat in args.categories.split(',')]
    
    # Initialize system
    try:
        system = ImageRecognitionSystem(
            device=args.device,
            ocr_languages=languages,
            custom_categories=custom_categories
        )
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False
    
    # Process images
    images_to_process = []
    
    if args.image:
        images_to_process.append(args.image)
    elif args.batch:
        images_to_process = [img.strip() for img in args.batch.split(',')]
    else:
        print("❌ No image specified. Use --image or --batch")
        return False
    
    # Validate images exist
    for img_path in images_to_process:
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            return False
    
    # Process each image
    all_results = []
    
    for img_path in images_to_process:
        if args.verbose:
            print(f"🔍 Processing: {img_path}")
        
        # Run analysis based on type
        if args.analysis_type == 'full':
            results = system.analyze_image(img_path)
        elif args.analysis_type == 'classification':
            results = {
                "image_path": img_path,
                "classification": system.classify_image(img_path),
                "analysis_time": 0
            }
        elif args.analysis_type == 'detection':
            results = {
                "image_path": img_path,
                "object_detection": system.detect_objects(img_path),
                "analysis_time": 0
            }
        elif args.analysis_type == 'caption':
            results = {
                "image_path": img_path,
                "caption": system.caption_image(img_path),
                "analysis_time": 0
            }
        elif args.analysis_type == 'ocr':
            results = {
                "image_path": img_path,
                "text_extraction": system.extract_text(img_path),
                "analysis_time": 0
            }
        
        all_results.append(results)
        
        # Print summary if verbose
        if args.verbose:
            system.print_analysis_summary(results)
    
    # Save results
    if args.output:
        if len(all_results) == 1:
            # Single result
            system.save_analysis(all_results[0], args.output)
        else:
            # Multiple results - save as array
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Batch results saved to: {args.output}")
    else:
        # Print JSON to stdout
        if len(all_results) == 1:
            print(json.dumps(all_results[0], indent=2, ensure_ascii=False))
        else:
            print(json.dumps(all_results, indent=2, ensure_ascii=False))
    
    return True


def main():
    """Main function to demonstrate the image recognition system."""
    args = parse_arguments()
    
    # Handle headless mode
    if args.headless:
        return run_headless_analysis(args)
    
    # Interactive mode
    print("🧠 TraeGPT Image Recognition System")
    print("="*50)
    
    # Initialize the system
    system = ImageRecognitionSystem()
    
    while True:
        print("\nOptions:")
        print("1. Analyze image")
        print("2. Classify image only")
        print("3. Detect objects only")
        print("4. Generate caption only")
        print("5. Extract text only")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "6":
            print("👋 Goodbye!")
            break
        
        image_path = input("Enter image path: ").strip()
        
        if not os.path.exists(image_path):
            print("❌ Image file not found!")
            continue
        
        if choice == "1":
            # Full analysis
            results = system.analyze_image(image_path)
            system.print_analysis_summary(results)
            
            # Save results
            save_choice = input("\nSave results to JSON? (y/n): ").strip().lower()
            if save_choice == 'y':
                output_path = f"analysis_{int(time.time())}.json"
                system.save_analysis(results, output_path)
        
        elif choice == "2":
            # Classification only
            results = system.classify_image(image_path)
            print("\n🏷️  CLASSIFICATION RESULTS:")
            for category, confidence in results.items():
                print(f"  • {category}: {confidence:.2%}")
        
        elif choice == "3":
            # Object detection only
            results = system.detect_objects(image_path)
            print("\n🎯 OBJECT DETECTION RESULTS:")
            for obj in results:
                print(f"  • {obj['class']}: {obj['confidence']:.2%}")
        
        elif choice == "4":
            # Caption only
            caption = system.caption_image(image_path)
            print(f"\n🖼️  CAPTION: {caption}")
        
        elif choice == "5":
            # Text extraction only
            results = system.extract_text(image_path)
            print("\n🔤 EXTRACTED TEXT:")
            for text_item in results:
                print(f"  • \"{text_item['text']}\" (confidence: {text_item['confidence']:.2%})")
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 