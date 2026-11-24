"""
Face Detection Service using OpenCV YuNet DNN
High-accuracy deep learning face detector with facial landmarks
Includes smoothing for stable, non-jittery detection
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

import config


@dataclass
class FaceDetection:
    """Represents a detected face"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: Optional[np.ndarray] = None
    
    @property
    def x(self) -> int:
        return self.bbox[0]
    
    @property
    def y(self) -> int:
        return self.bbox[1]
    
    @property
    def width(self) -> int:
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        return self.bbox[3]
    
    def get_center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def get_area(self) -> int:
        return self.width * self.height


class FaceDetector:
    """
    OpenCV YuNet DNN face detector.
    Uses deep learning for accurate, fast face detection with landmarks.
    Includes smoothing for stable detections.
    """
    
    def __init__(self):
        self._detector = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = False
        
        # YuNet parameters (configurable via config.py)
        self._score_threshold = getattr(config, 'YUNET_SCORE_THRESHOLD', 0.7)
        self._nms_threshold = getattr(config, 'YUNET_NMS_THRESHOLD', 0.3)
        self._input_size = getattr(config, 'YUNET_INPUT_SIZE', (320, 320))
        
        # Smoothing: track previous detections
        self._prev_detections: List[FaceDetection] = []
        self._last_detection_time = 0
        self._detection_timeout = 0.5  # seconds before clearing stale detections
        self._smoothing_factor = 0.4  # Blend factor for position smoothing (higher = more responsive)
        
    def initialize(self) -> bool:
        """Initialize the YuNet DNN face detector"""
        if self._initialized:
            return True
            
        try:
            model_path = str(config.YUNET_MODEL_PATH)
            if not config.YUNET_MODEL_PATH.exists():
                print(f"❌ YuNet model not found at {model_path}")
                print("   Run: python download_models.py")
                return False
            
            # Create YuNet detector
            self._detector = cv2.FaceDetectorYN_create(
                model_path,
                "",  # Config file (not needed)
                self._input_size,
                self._score_threshold,
                self._nms_threshold,
                5000  # Top K candidates
            )
            
            self._initialized = True
            print("✅ YuNet DNN face detector initialized")
            print(f"   Score threshold: {self._score_threshold}")
            print(f"   NMS threshold: {self._nms_threshold}")
            print(f"   Input size: {self._input_size}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize YuNet detector: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _smooth_detections(self, new_detections: List[FaceDetection]) -> List[FaceDetection]:
        """Smooth detection positions to reduce jitter"""
        current_time = time.time()
        
        # If too long since last detection, reset
        if current_time - self._last_detection_time > self._detection_timeout:
            self._prev_detections = []
        
        self._last_detection_time = current_time
        
        if not self._prev_detections or not new_detections:
            self._prev_detections = new_detections
            return new_detections
        
        # Match new detections to previous ones by proximity
        smoothed = []
        used_prev = set()
        
        for new_det in new_detections:
            new_center = new_det.get_center()
            best_match = None
            best_dist = float('inf')
            
            for i, prev_det in enumerate(self._prev_detections):
                if i in used_prev:
                    continue
                prev_center = prev_det.get_center()
                dist = ((new_center[0] - prev_center[0])**2 + (new_center[1] - prev_center[1])**2)**0.5
                
                # Only match if reasonably close
                if dist < max(new_det.width, new_det.height) * 1.5 and dist < best_dist:
                    best_dist = dist
                    best_match = i
            
            if best_match is not None:
                # Smooth the position
                prev = self._prev_detections[best_match]
                alpha = self._smoothing_factor
                
                smooth_x = int(prev.x * (1 - alpha) + new_det.x * alpha)
                smooth_y = int(prev.y * (1 - alpha) + new_det.y * alpha)
                smooth_w = int(prev.width * (1 - alpha) + new_det.width * alpha)
                smooth_h = int(prev.height * (1 - alpha) + new_det.height * alpha)
                
                smoothed.append(FaceDetection(
                    bbox=(smooth_x, smooth_y, smooth_w, smooth_h),
                    confidence=new_det.confidence,
                    landmarks=new_det.landmarks
                ))
                used_prev.add(best_match)
            else:
                # New face, no smoothing
                smoothed.append(new_det)
        
        self._prev_detections = smoothed
        return smoothed
    
    def detect(self, image: np.ndarray, apply_smoothing: bool = True) -> List[FaceDetection]:
        """
        Detect faces in an image using YuNet DNN.
        
        Args:
            image: BGR image as numpy array
            apply_smoothing: Whether to apply position smoothing
            
        Returns:
            List of FaceDetection objects with confidence and landmarks
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        if image is None or image.size == 0:
            return []
        
        h, w = image.shape[:2]
        
        # Update input size if image dimensions differ
        if (w, h) != self._input_size:
            self._detector.setInputSize((w, h))
        
        # Detect faces
        _, faces = self._detector.detect(image)
        
        if faces is None:
            return []
        
        detections = []
        for face in faces:
            # YuNet output: x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[14])
            
            # Extract 5-point landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner)
            landmarks = np.array([
                [face[4], face[5]],   # Right eye
                [face[6], face[7]],   # Left eye
                [face[8], face[9]],   # Nose tip
                [face[10], face[11]], # Right mouth corner
                [face[12], face[13]]  # Left mouth corner
            ], dtype=np.float32)
            
            detections.append(FaceDetection(
                bbox=(x, y, fw, fh),
                confidence=confidence,
                landmarks=landmarks
            ))
        
        # Apply smoothing to reduce jitter
        if apply_smoothing and detections:
            detections = self._smooth_detections(detections)
        
        return detections
    
    async def detect_async(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces asynchronously (non-blocking).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.detect, image)
    
    def draw_detections(self, image: np.ndarray, detections: List[FaceDetection], 
                        names: List[str] = None, statuses: List[str] = None) -> np.ndarray:
        """Draw detection boxes on image with names, status colors, and landmarks"""
        result = image.copy()
        
        for i, det in enumerate(detections):
            name = names[i] if names and i < len(names) else None
            status = statuses[i] if statuses and i < len(statuses) else "unknown"
            
            # Color based on status
            if status == "present":
                color = (0, 255, 0)  # Green for known
            elif status == "late":
                color = (0, 165, 255)  # Orange for late
            else:
                color = (0, 0, 255)  # Red for unknown
            
            # Draw bounding box
            x, y, w, h = det.x, det.y, det.width, det.height
            thickness = 2
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Draw confidence
            conf_text = f"{det.confidence:.0%}"
            cv2.putText(result, conf_text, (x + w - 40, y + h - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw landmarks if available
            if det.landmarks is not None:
                for lm in det.landmarks:
                    cv2.circle(result, (int(lm[0]), int(lm[1])), 2, (255, 255, 0), -1)
            
            # Draw name label if available
            if name:
                label = name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Draw label background
                cv2.rectangle(result, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
                
                # Draw text
                cv2.putText(result, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness)
        
        return result
    
    def extract_face(self, image: np.ndarray, detection: FaceDetection, 
                     padding: float = 0.2) -> Optional[np.ndarray]:
        """Extract face region from image with padding"""
        h, w = image.shape[:2]
        
        pad_w = int(detection.width * padding)
        pad_h = int(detection.height * padding)
        
        x1 = max(0, detection.x - pad_w)
        y1 = max(0, detection.y - pad_h)
        x2 = min(w, detection.x + detection.width + pad_w)
        y2 = min(h, detection.y + detection.height + pad_h)
        
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        return face
    
    def close(self):
        """Clean up resources"""
        self._executor.shutdown(wait=False)
        self._initialized = False
        self._prev_detections = []


# Global face detector instance
face_detector = FaceDetector()
