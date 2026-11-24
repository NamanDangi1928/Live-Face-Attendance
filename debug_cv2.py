import cv2

print(f"OpenCV Version: {cv2.__version__}")

try:
    print("cv2.face attributes:")
    print(dir(cv2.face))
except AttributeError:
    print("cv2.face does not exist")

print("\ntesting FaceRecognizerSF...")
try:
    if hasattr(cv2, 'FaceRecognizerSF_create'):
        print("Found cv2.FaceRecognizerSF_create")
    elif hasattr(cv2.face, 'FaceRecognizerSF_create'):
        print("Found cv2.face.FaceRecognizerSF_create")
    else:
        print("FaceRecognizerSF_create NOT FOUND")
except Exception as e:
    print(f"Error checking: {e}")
