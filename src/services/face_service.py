import cv2
import numpy as np
import os
import random

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize Haar Cascade Face Detection.
        """
        # Load the pre-trained Haar Cascade model from OpenCV's data
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
             cascade_path = 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.min_neighbors = 5
        self.scale_factor = 1.1
        self.min_size = (30, 30)
        
        # Load Deep Learning Models (Placeholder for Refactoring)
        # self.emotion_model = ...
        # self.gender_model = ...

    def detect_faces(self, image):
        """
        Detect faces and simulate AI analysis (Engagement, Emotion, Gender).
        Returns: 
        1. Annotated Image (with boxes drawn)
        2. List of dicts: [{'box':(x,y,w,h), 'emotion':'Happy', 'engagement':0.8, 'gender':'Male'}]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        annotated_frame = image.copy()
        
        for (x, y, w, h) in faces_rects:
            # 1. Draw Box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 2. Simulate AI Analysis (Since we are refactoring, we use placeholders 
            # or lightweight logic until deep learning models are re-wired)
            # In a real scenario, here we crop face -> feed to CNN -> get result.
            
            # Simulated Engagement (Random for demo, usually based on gaze)
            eng_score = random.uniform(0.5, 1.0) 
            
            # Simulated Emotion
            emotions = ['Happy', 'Neutral', 'Surprise', 'Sad']
            curr_emo = random.choice(emotions)
            
            # Simulated Gender
            genders = ['Male', 'Female']
            curr_gender = random.choice(genders)

            face_data = {
                'box': (x, y, w, h),
                'engagement': eng_score,
                'emotion': curr_emo,
                'gender': curr_gender
            }
            results.append(face_data)
            
            # Draw Labels
            label = f"{curr_emo} ({eng_score:.2f})"
            cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_frame, results

    def draw_detections(self, image, bboxes):
        for (x, y, w, h) in bboxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
