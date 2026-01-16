import sqlite3
import os
from datetime import datetime

class EngagementDB:
    def __init__(self, db_path="student_engagement.db"):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Khởi tạo cấu trúc bảng (Schema) chuẩn hóa."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Bảng 1: Sessions - Lưu thông tin tổng quan của một buổi học
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT,
                end_time TEXT,
                total_students INTEGER,
                avg_engagement REAL,
                class_status TEXT,
                male_ratio REAL,
                female_ratio REAL
            )
        ''')

        # Bảng 2: SessionLogs - Lưu dữ liệu time-series (từng thời điểm)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TEXT,
                face_count INTEGER,
                avg_engagement REAL,
                male_engagement REAL,
                female_engagement REAL,
                male_count INTEGER,
                female_count INTEGER,
                emotion_data TEXT, -- Lưu JSON string hoặc format dạng 'Happy:5,Sad:2'
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def start_session(self):
        """Tạo một session mới khi bắt đầu monitor."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('INSERT INTO sessions (start_time) VALUES (?)', (start_time,))
            session_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return session_id
        except Exception as e:
            print(f"[DB ERROR] Start Session failed: {e}")
            return None

    def log_frame_data(self, session_id, stats):
        """Ghi log chi tiết của từng frame (3-5s/lần)."""
        if not session_id: return
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Serialize emotion data to string
            emotions = []
            for k, v in stats.items():
                if k not in ['timestamp', 'face_count', 'avg_engagement', 'status', 'male_engagement', 'female_engagement', 'male_count', 'female_count', 'total_students']:
                    emotions.append(f"{k}:{v}")
            emotion_str = ",".join(emotions)

            cursor.execute('''
                INSERT INTO session_logs (
                    session_id, timestamp, face_count, avg_engagement, 
                    male_engagement, female_engagement, male_count, female_count, emotion_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, 
                stats.get('timestamp'), 
                stats.get('total_students', 0), 
                stats.get('avg_engagement', 0),
                stats.get('male_engagement', 0),
                stats.get('female_engagement', 0),
                stats.get('male_count', 0),
                stats.get('female_count', 0),
                emotion_str
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB ERROR] Log Frame failed: {e}")

    def end_session(self, session_id, summary_stats):
        """Cập nhật thông tin tổng kết khi kết thúc session."""
        if not session_id: return
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute('''
                UPDATE sessions 
                SET end_time = ?, 
                    total_students = ?, 
                    avg_engagement = ?, 
                    class_status = ?,
                    male_ratio = ?,
                    female_ratio = ?
                WHERE id = ?
            ''', (
                end_time,
                summary_stats.get('max_students', 0),
                summary_stats.get('avg_session_engagement', 0),
                summary_stats.get('final_status', 'N/A'),
                summary_stats.get('male_ratio', 0),
                summary_stats.get('female_ratio', 0),
                session_id
            ))
            conn.commit()
            conn.close()
            print(f"[DB SUCCESS] Session {session_id} saved successfully.")
        except Exception as e:
            print(f"[DB ERROR] End Session failed: {e}")

    def get_all_sessions(self):
        """Lấy lịch sử để hiển thị lên GUI."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM sessions ORDER BY id DESC')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_session_logs(self, session_id):
        """Lấy chi tiết logs của một session cụ thể để vẽ biểu đồ."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM session_logs WHERE session_id = ? ORDER BY id ASC', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows
