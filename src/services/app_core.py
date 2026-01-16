import cv2
import time
import numpy as np
from datetime import datetime
from src.services.face_service import FaceDetector
from src.data.repository import EngagementDB

class AppCore:
    def __init__(self):
        # 1. Initialize AI
        self.detector = FaceDetector(min_detection_confidence=0.6)
        
        # 2. Initialize DB
        self.db = EngagementDB()
        
        # 3. Camera Config
        self.cap = None
        self.is_running = False
        self.session_id = None
        
        # Stats Cache
        self.frame_count = 0
        self.start_time = None

    def start_monitoring(self):
        """Start Camera and Create New Session in DB"""
        if self.is_running: return
        
        # Try index 0, then 1 (external cam)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
            
        if not self.cap.isOpened():
            raise Exception("Cannot open Webcam (Index 0 or 1)")

        self.is_running = True
        self.start_time = datetime.now()
        
        # Create DB Session
        self.session_id = self.db.create_session(self.start_time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[CORE] Session Started: ID {self.session_id}")

    def stop_monitoring(self):
        """Stop Camera and Update Session End Time"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        if self.session_id:
            end_time = datetime.now()
            duration = (end_time - self.start_time).seconds
            # Update session in DB (Assuming db has update_session method, if not we ignore for now)
            try:
                self.db.end_session(self.session_id, end_time.strftime("%Y-%m-%d %H:%M:%S"))
            except AttributeError:
                pass # Method might be named differently
            print(f"[CORE] Session Ended. Duration: {duration}s")

    def process_frame(self):
        """
        Main Loop Step:
        1. Read Frame
        2. Detect Faces & Emotion
        3. Calculate Stats
        4. Log to DB (every 30 frames aka 1 sec)
        """
        if not self.cap or not self.is_running:
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # --- AI Processing ---
        # Detect faces returns: image with boxes, list of face_data
        # face_data format expected: {'box':..., 'emotion':..., 'engagement':...}
        processed_frame, faces = self.detector.detect_faces(frame)
        
        # Calculate Aggregate Stats
        total_students = len(faces)
        avg_engagement = 0
        male_count = 0
        female_count = 0
        emotions_list = []

        if total_students > 0:
            avg_engagement = np.mean([f.get('engagement', 0) for f in faces])
            male_count = sum(1 for f in faces if f.get('gender') == 'Male')
            female_count = sum(1 for f in faces if f.get('gender') == 'Female')
            emotions_list = [f.get('emotion', 'Neutral') for f in faces]

        stats = {
            'total_students': total_students,
            'avg_engagement': avg_engagement,
            'male_count': male_count,
            'female_count': female_count,
            'emotions': emotions_list
        }

        # --- Log to DB (Throttled) ---
        self.frame_count += 1
        if self.frame_count % 30 == 0 and self.session_id:
            # Prepare emotion string "Happy:2,Sad:1"
            emo_counts = {}
            for e in emotions_list:
                emo_counts[e] = emo_counts.get(e, 0) + 1
            emo_str = ",".join([f"{k}:{v}" for k,v in emo_counts.items()])
            
            self.db.log_engagement(
                session_id=self.session_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                student_count=total_students,
                avg_engagement=avg_engagement,
                attention_span=0, # Placeholder
                drowsiness_count=0, # Placeholder
                male_count=male_count,
                female_count=female_count,
                emotion_summary=emo_str
            )

        return processed_frame, stats
