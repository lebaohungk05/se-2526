import sys
import os
import unittest
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

class TestCoreIntegration(unittest.TestCase):
    def test_app_initialization(self):
        """Try to initialize the main core to check model loading."""
        print("\n[TEST] Attempting to import and init TeacherMonitorApp...")
        try:
            from teacher_tool.main_app import TeacherMonitorApp
            # This triggers __init__ which loads PyTorch, Keras, MediaPipe
            app = TeacherMonitorApp()
            print("[TEST] Initialization Successful.")
            
            # Dry run logic checks
            self.assertIsNotNone(app.emotion_model)
            self.assertIsNotNone(app.gender_model)
            self.assertIsNotNone(app.face_detector)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Failed to initialize Core App: {e}")

if __name__ == '__main__':
    unittest.main()

