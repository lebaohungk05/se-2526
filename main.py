import os
import sys

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui.main_window import ModernTeacherApp

if __name__ == "__main__":
    print("[INFO] Starting TeacherStationPro...")
    app = ModernTeacherApp()
    app.mainloop()
