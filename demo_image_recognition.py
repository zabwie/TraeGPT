#!/usr/bin/env python3
"""
Demo script for the TraeGPT Image Recognition System
This script demonstrates the capabilities of the image recognition system.
"""

import os
import sys
import argparse
import time
from image_recognition import ImageRecognitionSystem

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TraeGPT Image Recognition Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive demo
  python demo_image_recognition.py

  # Headless demo with custom settings
  python demo_image_recognition.py --headless --languages "en,es" --categories "person,dog,car"

  # Headless demo with specific image
  python demo_image_recognition.py --headless --image path/to/image.jpg
        """
    )
    
    # Mode arguments
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode without interactive prompts')
    parser.add_argument('--image', type=str, 
                       help='Path to specific image for demo (headless mode only)')
    
    # Configuration arguments
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                       help='Device to run models on (default: auto-detect)')
    parser.add_argument('--languages', type=str, default='en',
                       help='Comma-separated list of OCR languages (default: en)')
    parser.add_argument('--categories', type=str,
                       help='Comma-separated list of custom categories for CLIP classification')
    
    # Output arguments
    parser.add_argument('--output', type=str, 
                       help='Output file path for demo results (headless mode only)')
    
    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def run_headless_demo(args):
    """Run demo in headless mode."""
    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(',')]
    
    # Parse custom categories if provided
    custom_categories = None
    if args.categories:
        custom_categories = [cat.strip() for cat in args.categories.split(',')]
    
    # Initialize the system
    try:
        system = ImageRecognitionSystem(
            device=args.device,
            ocr_languages=languages,
            custom_categories=custom_categories
        )
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    
    # Find demo image
    demo_image = None
    
    if args.image:
        # Use specified image
        if os.path.exists(args.image):
            demo_image = args.image
        else:
            print(f"❌ Specified image not found: {args.image}")
            return False
    else:
        # Look for sample images
        sample_images = [
            "sample.jpg",
            "test.jpg", 
            "image.jpg",
            "demo.jpg"
        ]
        
        for img in sample_images:
            if os.path.exists(img):
                demo_image = img
                break
    
    if not demo_image:
        print("❌ No demo image found.")
        print("To test the system, place an image file in the current directory")
        print("or specify one with --image")
        return False
    
    if args.verbose:
        print(f"📸 Using demo image: {demo_image}")
    
    # Run full analysis
    print("\n🔍 Running full analysis...")
    results = system.analyze_image(demo_image)
    
    if "error" in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    # Print summary
    system.print_analysis_summary(results)
    
    # Save results if requested
    if args.output:
        system.save_analysis(results, args.output)
        print(f"💾 Demo results saved to: {args.output}")
    else:
        # Save with timestamp
        output_file = f"demo_analysis_{int(time.time())}.json"
        system.save_analysis(results, output_file)
        print(f"💾 Demo results saved to: {output_file}")
    
    return True


def demo_image_recognition():
    """Demonstrate the image recognition system capabilities."""
    print("🧠 TraeGPT Image Recognition Demo")
    print("="*50)
    
    # Initialize the system
    try:
        system = ImageRecognitionSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return
    
    # Demo with a sample image (if available)
    sample_images = [
        "sample.jpg",
        "test.jpg", 
        "image.jpg",
        "demo.jpg"
    ]
    
    # Check for sample images
    available_images = []
    for img in sample_images:
        if os.path.exists(img):
            available_images.append(img)
    
    if available_images:
        print(f"\n📸 Found sample images: {available_images}")
        demo_image = available_images[0]
        print(f"Using: {demo_image}")
        
        # Run full analysis
        print("\n🔍 Running full analysis...")
        results = system.analyze_image(demo_image)
        system.print_analysis_summary(results)
        
        # Save results
        output_file = f"demo_analysis_{int(time.time())}.json"
        system.save_analysis(results, output_file)
        
    else:
        print("\n📸 No sample images found.")
        print("To test the system, place an image file in the current directory")
        print("and run: python image_recognition.py")
    
    # Show system capabilities
    print("\n🎯 System Capabilities:")
    print("  • 🏷️  Image Classification (CLIP)")
    print("  • 🎯 Object Detection (YOLOv8)")
    print("  • 🖼️  Image Captioning (BLIP)")
    print("  • 🔤 Text Recognition (EasyOCR)")
    print("  • 💾 Results Export (JSON)")
    
    print("\n📝 Usage Examples:")
    print("  • python image_recognition.py")
    print("  • python ollama_enhanced.py")
    print("  • analyze: path/to/image.jpg")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Handle headless mode
    if args.headless:
        return run_headless_demo(args)
    
    # Interactive demo
    demo_image_recognition()
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 