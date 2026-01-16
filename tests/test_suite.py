import unittest
import sys
import os
import sqlite3
import numpy as np

# Add src path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from teacher_tool.database import EngagementDB
# Giả lập logic tính điểm (vì nó nằm trong main_app.py khó import lẻ, 
# ta sẽ test logic công thức tại đây để đảm bảo tính đúng đắn của thuật toán)

def calculate_engagement(emotions):
    # Copy of the logic in main_app.py
    # Score = 0.5*Neutral + 1.0*Happy + 1.0*Surprise - 0.5*Negative
    weights = {
        'Happy': 1.0, 'Surprise': 1.0, 'Neutral': 0.5,
        'Sad': -0.5, 'Angry': -0.5, 'Fear': -0.5, 'Disgust': -0.5
    }
    score = 0
    total_count = sum(emotions.values())
    if total_count == 0: return 0
    
    for emo, count in emotions.items():
        score += count * weights.get(emo, 0)
    
    return score / total_count

class TestEngagementSystem(unittest.TestCase):

    def test_engagement_algorithm_positive(self):
        """Test case 1: Lớp học vui vẻ -> Điểm phải cao"""
        emotions = {'Happy': 10, 'Neutral': 5, 'Sad': 0}
        # Score = (10*1.0 + 5*0.5) / 15 = 12.5 / 15 = 0.833
        score = calculate_engagement(emotions)
        self.assertGreater(score, 0.5)
        print(f"Positive Class Score: {score:.3f}")

    def test_engagement_algorithm_negative(self):
        """Test case 2: Lớp học chán nản -> Điểm phải thấp/âm"""
        emotions = {'Happy': 0, 'Neutral': 2, 'Sad': 8, 'Angry': 2}
        # Score = (2*0.5 + 8*-0.5 + 2*-0.5) / 12 = (1 - 4 - 1) / 12 = -4/12 = -0.33
        score = calculate_engagement(emotions)
        self.assertLess(score, 0)
        print(f"Negative Class Score: {score:.3f}")

    def test_database_operations(self):
        """Test case 3: Kiểm tra CRUD Database (Sử dụng file tạm)"""
        test_db_path = "test_engagement.db"
        if os.path.exists(test_db_path): os.remove(test_db_path)
        
        db = EngagementDB(test_db_path)
        
        # 1. Start Session
        session_id = db.start_session()
        self.assertIsNotNone(session_id)
        
        # 2. Log Data
        sample_stats = {
            'timestamp': '12:00:01', 'total_students': 20, 
            'avg_engagement': 0.75, 'male_count': 10, 'female_count': 10,
            'Happy': 15, 'Neutral': 5
        }
        db.log_frame_data(session_id, sample_stats)
        
        # 3. End Session
        summary = {
            'max_students': 20, 'avg_session_engagement': 0.7, 
            'final_status': 'HIGH FOCUS', 'male_ratio': 0.5, 'female_ratio': 0.5
        }
        db.end_session(session_id, summary)
        
        # 4. Verify Data retrieval
        sessions = db.get_all_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0][3], 20) # total_students column
        
        # Cleanup
        conn = db.get_connection()
        conn.close()
        os.remove(test_db_path)
        print("Database CRUD Test Passed.")

if __name__ == '__main__':
    unittest.main()
