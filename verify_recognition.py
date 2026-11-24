import asyncio
import cv2
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from database import mongodb
from services.face_recognizer import face_recognizer
import config

async def test_recognition():
    print("----- STARTING RECOGNITION TEST -----")
    
    # 1. Connect to DB
    print("1. Connecting to MongoDB...")
    try:
        if not await mongodb.connect_mongodb():
            print("❌ MongoDB Connection Failed")
            return
            
        # 2. Initialize Recognizer
        print("\n2. Initializing Face Recognizer...")
        # Face recognizer initializes in a fire-and-forget task in _load_enrolled_faces
        # We need to manually trigger load or wait
        success = face_recognizer.initialize()
        if not success:
            print("❌ Initialization failed")
            return
            
        # Wait for potential background loading
        print("   Waiting for background loading...")
        await asyncio.sleep(2)
        
        count = face_recognizer.get_enrolled_count()
        print(f"\n✅ Enrolled count: {count}")
        
        if count == 0:
            print("❌ No students enrolled. Cannot test recognition.")
            # return # Continue to try to see errors, but realistically fail
        
        # 3. Test with images
        images = list(config.STUDENTS_DIR.glob("*.jpg"))
        if not images:
            print("❌ No images found on disk.")
            return
            
        test_image_path = images[0]
        print(f"\n3. Testing with image: {test_image_path.name}")
        
        img = cv2.imread(str(test_image_path))
        if img is None:
            print("❌ Could not read test image")
            return
            
        # Run recognition
        # NOTE: ensure we await the async wrapper
        match = await face_recognizer.recognize_async(img)
        
        print("\n----- RESULT -----")
        if match:
            print("✅ MATCH FOUND!")
            print(f"Name: {match.name}")
            print(f"Similarity: {match.similarity:.2f}")
            print(f"Student ID: {match.student_id}")
        else:
            print("❌ NO MATCH FOUND (False Negative)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n4. Cleaning up...")
        face_recognizer.close()
        await mongodb.close_mongodb()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(test_recognition())
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
