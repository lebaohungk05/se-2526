# Teacher Station Pro - Student Engagement Analysis System

## 📋 Overview
**Teacher Station Pro** is an AI-powered desktop application designed to help educators monitor and analyze student engagement in real-time. By utilizing computer vision and deep learning, the system analyzes facial expressions via a webcam feed to estimate the overall "focus score" of the classroom without recording or storing video data, ensuring privacy.

The application features a modern, responsive dashboard built with `CustomTkinter` and utilizes a hybrid AI backend combining **PyTorch** (for Emotion Recognition) and **Keras/TensorFlow** (for Gender Classification).

## ✨ Key Features
*   **Real-time Monitoring:** Live detection of student faces, emotions, and gender.
*   **Engagement Scoring:** Calculates a weighted "Focus Score" based on positive/negative emotional cues (e.g., Happy/Surprise vs. Sad/Bored).
*   **Analytics Dashboard:** Visualizes trends, gender distribution, and dominant emotions during a session.
*   **Session History:** Archives session data into a local SQLite database for post-class review.
*   **Modern UI:** A clean, dark/light mode capable interface using CustomTkinter.
*   **Local Processing:** All AI processing runs locally on the machine. No data is sent to the cloud.

## 🛠️ Tech Stack
*   **Language:** Python 3.8+
*   **GUI:** CustomTkinter (Modern wrapper for Tkinter)
*   **Computer Vision:** OpenCV (Face detection, Image processing)
*   **AI/Deep Learning:**
    *   **Emotion Model:** PyTorch (ResNet18 architecture trained on FER2013).
    *   **Gender Model:** TensorFlow/Keras (Mini_XCEPTION architecture).
*   **Database:** SQLite (via Python `sqlite3`).
*   **Packaging:** PyInstaller.

## 📂 Project Structure
```bash
D:\SE\
├── datasets/                # Raw data for training (ignored in git)
├── trained_models/          # Pre-trained model weights (.pth, .hdf5)
├── reports/                 # Exported CSV session reports
├── src/                     # Source Code
│   ├── teacher_tool/        # MAIN APPLICATION LOGIC
│   │   ├── main_app_modern.py  # <--- ENTRY POINT (Run this)
│   │   ├── main_app.py         # Backend logic (AI integration, Threads)
│   │   ├── dashboard.py        # UI Components
│   │   └── database.py         # SQLite interactions
│   ├── models/              # Neural Network Architectures
│   │   ├── resnet18_pytorch.py # Current Emotion Model
│   │   └── cnn.py              # Current Gender Model (Mini_XCEPTION)
│   ├── utils/               # Helper functions (Pre-processing, plotting)
│   └── train_*.py           # Training scripts for different models
├── TeacherStationPro.spec   # PyInstaller build configuration
└── REQUIREMENTS.txt         # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.9 or higher.
*   A webcam connected to the computer.
*   (Optional) NVIDIA GPU with CUDA for faster training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd lms-app
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r REQUIREMENTS.txt
    ```
    *Note: If you have issues with `dlib` or `torch`, install them specifically for your hardware configuration.*

4.  **Verify Model Files:**
    Ensure the `trained_models/` directory contains:
    *   `emotion_models/fer2013_resnet18_best_sgd.pth`
    *   `gender_models/gender_mini_XCEPTION.21-0.95.hdf5`

## 🖥️ Usage

**Run the Application:**
```bash
python src/teacher_tool/main_app_modern.py
```

1.  **Dashboard:** Click "Start Session" to begin monitoring.
2.  **Live Feed:** The camera will detect faces. Green boxes indicate high engagement, Red/Blue indicate lower engagement.
3.  **Analytics:** Switch tabs to view real-time graphs.
4.  **Stop:** Click "Stop Session" to save data to the local database.

## 🧠 Development & Training

### Retraining Emotion Model (PyTorch)
To train a new emotion classifier using the FER2013 dataset:
1.  Place data in `datasets/fer2013/`.
2.  Run:
    ```bash
    python src/train_emotion_classifier_mobilenet_pytorch.py
    ```
    *(Note: The script name mentions MobileNet, but check the import to ensure you are training the desired architecture).*

### Building the Executable (.exe)
To package the app for distribution (Windows):
```bash
pyinstaller TeacherStationPro.spec
```
The output will be in the `dist/` folder.

## 🤝 Contribution Guidelines
1.  **Fork** the project.
2.  Create a **Feature Branch** (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes.
4.  **Push** to the branch.
5.  Open a **Pull Request**.

## 📄 License
[Insert License Here]
