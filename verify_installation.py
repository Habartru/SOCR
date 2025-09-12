#!/usr/bin/env python3
"""
Script to verify that all dependencies are properly installed
"""

def verify_installation():
    """Verify that all required dependencies are installed"""
    required_packages = [
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
        'PIL',  # pillow
        'requests',
        'cv2',
        'pypdfium2',
        'fitz',  # pymupdf
        'dotenv',
        'huggingface_hub',
        'packaging',
        'yaml',
        'regex',
        'safetensors',
        'tokenizers',
        'distlib',
        'platformdirs',
        'click',
        'einops',
        'transformers',
        'timm',
        'pydantic',
        'pydantic_settings',
        'surya',
        'setuptools',
        'wheel',
        'tiktoken',
        'filetype'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ {package} (OpenCV) - version: {cv2.__version__}")
            elif package == 'fitz':
                import fitz
                # PyMuPDF doesn't have __version__ attribute, but we can check if it's imported
                print(f"✅ {package} (PyMuPDF) - imported successfully")
            elif package == 'PIL':
                from PIL import Image
                # PIL is part of pillow package
                print(f"✅ {package} (Pillow) - imported successfully")
            elif package == 'yaml':
                import yaml
                print(f"✅ {package} (PyYAML) - version: {yaml.__version__}")
            elif package == 'dotenv':
                from dotenv import load_dotenv
                print(f"✅ {package} (python-dotenv) - imported successfully")
            elif package == 'surya':
                from surya.input.load import load_from_file
                from surya.detection import DetectionPredictor
                from surya.recognition import RecognitionPredictor
                from surya.foundation import FoundationPredictor
                print(f"✅ {package} - all components imported successfully")
            elif package == 'pydantic_settings':
                from pydantic_settings import BaseSettings
                print(f"✅ {package} - imported successfully")
            else:
                __import__(package)
                print(f"✅ {package} - imported successfully")
        except ImportError as e:
            print(f"❌ {package} - failed to import: {e}")
            missing_packages.append(package)
        except Exception as e:
            print(f"❌ {package} - error: {e}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n🎉 All packages are installed correctly!")
        return True

if __name__ == "__main__":
    print("Verifying installation...")
    print("=" * 40)
    success = verify_installation()
    
    if success:
        print("\n✅ Installation verification passed!")
        print("The application should work without import errors.")
    else:
        print("\n❌ Installation verification failed!")
        print("Please check your installation and try again.")