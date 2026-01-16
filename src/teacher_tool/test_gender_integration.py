import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model

# Setup paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from teacher_tool.face_detector import FaceDetector

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5')
TEST_IMAGE_PATH = os.path.join(BASE_DIR, 'images/test_image.jpg') # Cần đảm bảo ảnh này tồn tại hoặc dùng ảnh khác

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def test_gender_model():
    print(f"[TEST] Loading Model: {os.path.basename(MODEL_PATH)}")
    try:
        gender_model = load_model(MODEL_PATH, compile=False)
        input_shape = gender_model.input_shape[1:3] # (48, 48) usually
        print(f"[TEST] Model Input Shape: {input_shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print("[TEST] Loading Detector...")
    detector = FaceDetector()
    
    # Check test image
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[WARN] Test image not found at {TEST_IMAGE_PATH}. Using a dummy black image.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a white circle to simulate a face
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1) 
    else:
        frame = cv2.imread(TEST_IMAGE_PATH)
    
    bboxes, _ = detector.detect_faces(frame)
    print(f"[TEST] Faces detected: {len(bboxes)}")
    
    for i, (x, y, w, h) in enumerate(bboxes):
        face_rgb = frame[y:y+h, x:x+w]
        # Convert BGR (OpenCV) to RGB (Model expects RGB based on demo code analysis)
        face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
        
        try:
            face_resized = cv2.resize(face_rgb, input_shape)
        except:
            continue
            
        # Preprocess (check utils/preprocessor.py logic later, but assuming simple scaling for now based on demo)
        # Demo code used: preprocess_input(x, False) -> x/255.0
        # Let's try standard normalization [0,1] first
        face_arr = face_resized.astype("float32") / 255.0
        face_arr = np.expand_dims(face_arr, axis=0)
        
        preds = gender_model.predict(face_arr, verbose=0)[0]
        label_idx = np.argmax(preds)
        # Labels usually: 0=Woman, 1=Man for IMDB datasets, but let's check probability
        
        gender = "Woman" if label_idx == 0 else "Man"
        print(f"Face {i+1}: Prediction={preds} -> {gender} (Index {label_idx})")

if __name__ == "__main__":
    test_gender_model()
