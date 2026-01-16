import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize Haar Cascade Face Detection.
        Fallback from MediaPipe due to Protobuf/TensorFlow version conflicts on Windows.
        """
        # Load the pre-trained Haar Cascade model from OpenCV's data
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
             # Fallback: try to find it in local directory or common paths if cv2.data is wrong
             cascade_path = 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print(f"[ERROR] Failed to load Haar Cascade from {cascade_path}")
        else:
            print(f"[INFO] Haar Cascade Face Detector initialized. Path: {cascade_path}")

        self.min_neighbors = 5
        self.scale_factor = 1.1
        self.min_size = (30, 30)

    def detect_faces(self, image):
        """
        Detect faces using Haar Cascade.
        :param image: BGR image from OpenCV.
        :return: List of bounding boxes [(x, y, w, h), ...], raw_results (empty for Haar)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples and ensure int types
        bboxes = []
        for (x, y, w, h) in faces:
            bboxes.append((int(x), int(y), int(w), int(h)))
                
        return bboxes, []

    def draw_detections(self, image, bboxes):
        """
        Draw bounding boxes manually.
        """
        for (x, y, w, h) in bboxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
