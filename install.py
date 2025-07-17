#!/usr/bin/env python3
"""
Installation script for TraeGPT Image Recognition System
Automates the setup process for all dependencies.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def install_dependencies():
    """Install all required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    core_packages = [
        "torch",
        "torchvision", 
        "ftfy",
        "regex",
        "tqdm",
        "pillow",
        "numpy",
        "opencv-python",
        "requests"
    ]
    
    for package in core_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            return False
    
    # Install CLIP
    if not run_command(f"{sys.executable} -m pip install git+https://github.com/openai/CLIP.git", "Installing CLIP"):
        return False
    
    # Install transformers and accelerate
    if not run_command(f"{sys.executable} -m pip install transformers accelerate", "Installing transformers"):
        return False
    
    # Install BLIP
    if not run_command(f"{sys.executable} -m pip install salesforce-blip", "Installing BLIP"):
        return False
    
    # Install YOLOv8
    if not run_command(f"{sys.executable} -m pip install ultralytics", "Installing YOLOv8"):
        return False
    
    # Install EasyOCR
    if not run_command(f"{sys.executable} -m pip install easyocr", "Installing EasyOCR"):
        return False
    
    return True

def test_imports():
    """Test if all modules can be imported successfully."""
    print("\n🧪 Testing imports...")
    
    test_modules = [
        "torch",
        "torchvision",
        "clip",
        "ftfy",
        "regex",
        "tqdm",
        "PIL",
        "numpy",
        "cv2",
        "transformers",
        "ultralytics",
        "easyocr"
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        return False
    
    print("✅ All modules imported successfully!")
    return True

def check_ollama():
    """Check if Ollama is available."""
    print("\n🤖 Checking Ollama...")
    
    try:
        # Try to connect to Ollama
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print("⚠️  Ollama server responded but with unexpected status")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server is not running")
        print("Please start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def main():
    """Main installation function."""
    print("🧠 TraeGPT Image Recognition System - Installation")
    print("="*60)
    
    # Check system requirements
    if not check_python_version():
        return False
    
    if not check_pip():
        print("Please install pip first")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the error messages above.")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Some dependencies may not be installed correctly.")
        return False
    
    # Check Ollama
    ollama_available = check_ollama()
    
    print("\n" + "="*60)
    print("🎉 Installation Summary")
    print("="*60)
    
    if ollama_available:
        print("✅ All components installed and ready!")
        print("\n🚀 You can now run:")
        print("  • python ollama_enhanced.py - Enhanced chat with image recognition")
        print("  • python image_recognition.py - Standalone image recognition")
        print("  • python demo_image_recognition.py - Demo the system")
    else:
        print("⚠️  Dependencies installed but Ollama is not running")
        print("\nTo start Ollama:")
        print("  1. Install Ollama from https://ollama.com/download")
        print("  2. Run: ollama serve")
        print("  3. Pull the model: ollama pull HammerAI/mistral-nemo-uncensored")
    
    print("\n📚 For more information, see README.md")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 