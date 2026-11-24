import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent dir to path to import config
sys.path.insert(0, str(Path(__file__).parent))
try:
    import config
    print(f"SFace Path from config: {config.SFACE_MODEL_PATH}")
except ImportError:
    print("Could not import config")
    # Fallback default
    config = None

def test_sface():
    print("--- Testing SFace Embedding ---")
    
    # 1. Load Model
    model_path = "models/face_recognition_sface_2021dec.onnx"
    if not Path(model_path).exists():
        print(f"❌ Model file missing at {model_path}")
        return

    try:
        if hasattr(cv2, 'FaceRecognizerSF_create'):
            rec = cv2.FaceRecognizerSF_create(model_path, "")
        elif hasattr(cv2.face, 'FaceRecognizerSF_create'):
            rec = cv2.face.FaceRecognizerSF_create(model_path, "")
        else:
            print("❌ FaceRecognizerSF_create not found")
            return
        
        print("✅ Recognizer created")

        # 2. Create Dummy Image (Face-like structure not needed for raw execution test)
        # SFace input is 112x112 BGR
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        # Draw a circle so it's not empty
        cv2.circle(img, (56, 56), 30, (255, 255, 255), -1)
        
        print(f"Input shape: {img.shape}, dtype: {img.dtype}")

        # 3. Compute Feature (Pre-aligned is False? feature() assumes aligned usually)
        # But let's try calling it directly.
        try:
            print("Calling feature()...")
            emb = rec.feature(img)
            
            if emb is not None:
                print(f"✅ Embedding Success! Shape: {emb.shape}")
                print(f"Sample: {emb.flatten()[:5]}")
            else:
                print("❌ Feature returned None")
                
        except cv2.error as e:
            print(f"❌ OpenCV Error: {e}")
        except Exception as e:
            print(f"❌ Python Error: {e}")

    except Exception as e:
        print(f"❌ Initialization Error: {e}")

if __name__ == "__main__":
    test_sface()
