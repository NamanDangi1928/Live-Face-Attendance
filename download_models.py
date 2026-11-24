"""
Model Download Script for Smart Attendance System
Downloads YuNet and prepares InsightFace models for offline use
"""
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import urllib.request
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, YUNET_MODEL_PATH, SFACE_MODEL_PATH


def download_file(url: str, dest: Path, description: str = ""):
    """Download a file with progress indicator"""
    print(f"\n📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Destination: {dest}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r   Progress: {percent}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest, progress_hook)
        print("\n   ✅ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        return False


def download_yunet():
    """Download YuNet face detection model"""
    if YUNET_MODEL_PATH.exists():
        print(f"✅ YuNet model already exists at {YUNET_MODEL_PATH}")
        return True
    
    # YuNet model URL from OpenCV Zoo
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    
    return download_file(url, YUNET_MODEL_PATH, "YuNet Face Detection Model")


def download_sface():
    """Download SFace face recognition model (OpenCV Zoo)"""
    if SFACE_MODEL_PATH.exists():
        print(f"✅ SFace model already exists at {SFACE_MODEL_PATH}")
        return True
    
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    return download_file(url, SFACE_MODEL_PATH, "SFace Recognition Model")


def setup_insightface():
    """Setup InsightFace models"""
    print("\n📦 Setting up InsightFace...")
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        print("   Downloading InsightFace buffalo_l model...")
        print("   (This may take a few minutes on first run)")
        
        # This will automatically download the model
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS_DIR),
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))
        
        print("   ✅ InsightFace models ready!")
        return True
        
    except ImportError:
        print("   ⚠️ InsightFace not installed. Install with: pip install insightface")
        return False
    except Exception as e:
        print(f"   ❌ Error setting up InsightFace: {e}")
        return False


def main():
    print("="*50)
    print("🤖 Smart Attendance System - Model Setup")
    print("="*50)
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Download YuNet
    yunet_ok = download_yunet()
    
    # Download SFace
    sface_ok = download_sface()
    
    # Setup InsightFace
    insightface_ok = setup_insightface()
    
    # Summary
    print("\n" + "="*50)
    print("📊 Setup Summary:")
    print(f"   YuNet: {'✅ Ready' if yunet_ok else '❌ Failed'}")
    print(f"   SFace: {'✅ Ready' if sface_ok else '❌ Failed'}")
    print(f"   InsightFace: {'✅ Ready' if insightface_ok else '❌ Failed'}")
    print("="*50)
    
    if yunet_ok and sface_ok:
        print("\n🎉 All models ready! You can now run the server.")
        print("   Command: python app.py")
    else:
        print("\n⚠️ Some models failed to download. Check errors above.")


if __name__ == "__main__":
    main()
