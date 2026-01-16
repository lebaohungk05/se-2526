import os
import cv2
import numpy as np
import mss
import time
import pandas as pd
from datetime import datetime
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
import tensorflow as tf
from tensorflow.keras.models import load_model

# Adjusting python path to include src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from teacher_tool.face_detector import FaceDetector
from teacher_tool.screen_capture import ScreenCapture
from teacher_tool.database import EngagementDB
from models.resnet18_pytorch import resnet18_pytorch

# --- CONFIGURATION ---
def get_base_path():
    """Returns the base path of the project, working both in dev and frozen exe."""
    if getattr(sys, 'frozen', False):
        # Running as compiled .exe (PyInstaller)
        # PyInstaller v6+ puts dependencies/datas in _internal in onedir mode
        base_dir = os.path.dirname(sys.executable)
        internal_dir = os.path.join(base_dir, '_internal')
        if os.path.exists(internal_dir):
            return internal_dir
        return base_dir
    else:
        # Running from source code
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(script_dir, '..', '..'))

BASE_PROJECT_PATH = get_base_path()

# Model Paths
EMOTION_MODEL_PATH = os.path.join(BASE_PROJECT_PATH, 'trained_models', 'emotion_models', 'fer2013_resnet18_best_sgd.pth')
GENDER_MODEL_PATH = os.path.join(BASE_PROJECT_PATH, 'trained_models', 'gender_models', 'gender_mini_XCEPTION.21-0.95.hdf5')
DB_PATH = os.path.join(BASE_PROJECT_PATH, 'student_engagement.db')

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
GENDER_LABELS = ['Female', 'Male']
ENGAGEMENT_WEIGHTS = {
    'Happy': 1.0, 'Surprise': 1.0, 'Neutral': 0.5,
    'Sad': -0.5, 'Angry': -0.5, 'Fear': -0.5, 'Disgust': -0.5
}

class TeacherMonitorApp:
    def __init__(self):
        print("[INIT] Setting up AI Environment (PyTorch + TensorFlow)...")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load ResNet18 (PyTorch) for Emotions
        print(f"[INIT] Loading ResNet18 Emotion Model from: {os.path.basename(EMOTION_MODEL_PATH)}")
        self.emotion_model = resnet18_pytorch(num_classes=7, pretrained=False)
        self.emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=self.device))
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
        
        # Emotion Preprocessing
        self.emotion_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 2. Load Mini XCEPTION (Keras) for Gender
        print(f"[INIT] Loading Gender Model...")
        self.gender_model = load_model(GENDER_MODEL_PATH, compile=False)
        self.gender_target_size = self.gender_model.input_shape[1:3]
        
        # 3. Face Detector (Optimized Haar Cascade)
        print("[INIT] Initializing Face Detector...")
        self.face_detector = FaceDetector() # Haar uses internal optimized params
        
        # 4. Database & Screen
        print(f"[INIT] Connecting to Database at: {DB_PATH}")
        self.db = EngagementDB(DB_PATH)
        self.screen_cap = ScreenCapture()
        self.is_running = False
        self.current_session_id = None
        
        # Optimization
        self.frame_count = 0
        self.skip_frames = 2 # Reduced skip because ResNet is fast on CPU/GPU
        self.face_cache = []
        
        # Session Stats
        self.session_stats_accumulator = {
            'total_engagement_sum': 0, 'count_frames': 0, 'max_students': 0,
            'male_counts': 0, 'female_counts': 0
        }
        print("[INIT] System Ready.")

    def start_monitoring(self):
        self.is_running = True
        self.current_session_id = self.db.start_session()
        self.session_stats_accumulator = {k: 0 for k in self.session_stats_accumulator}
        print(f"[INFO] Session Started (ID: {self.current_session_id})")

    def stop_monitoring(self):
        self.is_running = False
        if self.current_session_id:
            acc = self.session_stats_accumulator
            avg_eng = acc['total_engagement_sum'] / acc['count_frames'] if acc['count_frames'] > 0 else 0
            total_gender = acc['male_counts'] + acc['female_counts']
            male_ratio = acc['male_counts'] / total_gender if total_gender > 0 else 0
            female_ratio = acc['female_counts'] / total_gender if total_gender > 0 else 0
            status = "HIGH FOCUS" if avg_eng > 0.3 else "NEUTRAL" if avg_eng > -0.2 else "DISTRACTED"
            
            summary = {
                'max_students': acc['max_students'],
                'avg_session_engagement': avg_eng,
                'final_status': status,
                'male_ratio': male_ratio,
                'female_ratio': female_ratio
            }
            self.db.end_session(self.current_session_id, summary)
            self.current_session_id = None

    def preprocess_gender(self, face_img):
        try:
            roi = cv2.resize(face_img, self.gender_target_size)
            roi = roi.astype("float32") / 255.0
            roi = (roi - 0.5) * 2.0
            roi = np.expand_dims(roi, axis=0)
            if self.gender_model.input_shape[-1] == 1:
                roi = cv2.cvtColor(roi[0], cv2.COLOR_BGR2GRAY)
                roi = np.expand_dims(np.expand_dims(roi, 0), -1)
            return roi
        except: return None

    def process_frame(self):
        if not self.is_running: return None, None
        frame = self.screen_cap.capture()
        if frame is None: return None, None

        self.frame_count += 1
        run_ai = (self.frame_count % self.skip_frames == 0)

        bboxes, _ = self.face_detector.detect_faces(frame)
        
        total_eng = {'Male': 0, 'Female': 0, 'All': 0}
        count = {'Male': 0, 'Female': 0, 'All': 0}
        current_emotions = {label: 0 for label in EMOTION_LABELS}
        new_cache = []
        
        for (x, y, w, h) in bboxes:
            h_img, w_img = frame.shape[:2]
            x, y = max(0, x), max(0, y)
            w, h = min(w, w_img - x), min(h, h_img - y)
            if w <= 10 or h <= 10: continue 
            
            face_img = frame[y:y+h, x:x+w]
            
            # 1. Gender (Keras)
            g_input = self.preprocess_gender(face_img)
            gender = "Male"
            if g_input is not None:
                g_preds = self.gender_model.predict(g_input, verbose=0)[0]
                gender = GENDER_LABELS[np.argmax(g_preds)]
            
            # 2. Emotion (PyTorch ResNet18)
            emotion = "Neutral"
            score = 0.5
            if run_ai:
                with torch.no_grad():
                    input_tensor = self.emotion_transform(face_img).unsqueeze(0).to(self.device)
                    outputs = self.emotion_model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    label_idx = torch.argmax(probs, dim=1).item()
                    emotion = EMOTION_LABELS[label_idx]
                    score = probs[0][label_idx].item()
            
            # Stats update
            eng_val = ENGAGEMENT_WEIGHTS.get(emotion, 0)
            count['All'] += 1
            total_eng['All'] += eng_val
            count[gender] += 1
            total_eng[gender] += eng_val
            current_emotions[emotion] += 1
            
            # Draw
            box_color = (255, 100, 0) if gender == 'Male' else (255, 100, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, f"{gender[0]}-{emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        stats = None
        if count['All'] > 0:
            avg_eng_all = total_eng['All'] / count['All']
            avg_eng_male = total_eng['Male'] / count['Male'] if count['Male'] > 0 else 0
            avg_eng_female = total_eng['Female'] / count['Female'] if count['Female'] > 0 else 0
            status = "HIGH FOCUS" if avg_eng_all > 0.3 else "NEUTRAL" if avg_eng_all > -0.2 else "DISTRACTED"
            
            stats = {
                "timestamp": datetime.now().strftime('%H:%M:%S'), 
                "total_students": count['All'],
                "avg_engagement": avg_eng_all,
                "male_engagement": avg_eng_male,
                "female_engagement": avg_eng_female,
                "male_count": count['Male'],
                "female_count": count['Female'],
                "status": status
            }
            stats.update(current_emotions)
            
            # Log more frequently for testing/demo (every 5 frames ~ 0.2s - 0.5s)
            # In production, use time-based logging (e.g., every 1-2 seconds)
            if self.frame_count % 5 == 0:
                self.db.log_frame_data(self.current_session_id, stats)
            
            self.session_stats_accumulator['total_engagement_sum'] += avg_eng_all
            self.session_stats_accumulator['count_frames'] += 1
            self.session_stats_accumulator['max_students'] = max(self.session_stats_accumulator['max_students'], count['All'])
            self.session_stats_accumulator['male_counts'] += count['Male']
            self.session_stats_accumulator['female_counts'] += count['Female']

        return frame, stats

if __name__ == "__main__":
    app = TeacherMonitorApp()
    # (Simplified CLI test runner if needed)
