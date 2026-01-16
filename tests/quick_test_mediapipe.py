import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from teacher_tool.face_detector import FaceDetector
    print("[TEST] Importing FaceDetector... SUCCESS")
    
    detector = FaceDetector()
    print("[TEST] Initialization... SUCCESS")
    
    # Create dummy black image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a white rectangle to simulate a "face" (MediaPipe won't detect it, but code should run)
    cv2.rectangle(dummy_img, (100, 100), (200, 200), (255, 255, 255), -1)
    
    print("[TEST] Running detection on dummy image...")
    bboxes, _ = detector.detect_faces(dummy_img)
    print(f"[TEST] Detection run complete. Found {len(bboxes)} faces (Expected 0 for dummy).")
    
    print("[TEST] ALL SYSTEMS GO. MediaPipe is working.")
except Exception as e:
    print(f"[TEST] FAILED with error: {e}")
