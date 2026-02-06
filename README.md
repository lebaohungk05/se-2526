# Teacher Station Pro

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Teacher Station Pro** is an intelligent desktop application designed for modern classrooms. It leverages **AI and Computer Vision** to analyze student engagement and emotional states in real-time, providing educators with actionable insights to improve teaching effectiveness.

> **Note:** This project is a software engineering coursework deliverable, demonstrating a full integration of Deep Learning models into a production-ready desktop application.

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Architecture](#overall-architecture)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Folder Structure](#folder-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

In large classrooms or online sessions, it is challenging for instructors to gauge the engagement level of every student simultaneously. **Teacher Station Pro** solves this by acting as a smart assistant that:
1.  **Observes** the class via camera feeds.
2.  **Analyzes** facial expressions and attention spans using lightweight Deep Learning models.
3.  **Reports** aggregated metrics (Engagement Score, Emotion Trends) via a live dashboard.

The system prioritizes **privacy**: no video footage is recorded. Only anonymous statistical data is processed and stored locally.

---

## Key Features

-   **📸 Real-time Monitoring**: Connects to webcams or external cameras to detect faces and analyze expressions on the fly (15+ FPS).
-   **🧠 Engagement Scoring**: Calculates a dynamic "Engagement Score" (0.0 - 1.0) based on weighted emotional indicators (e.g., Happy/Neutral vs. Bored/Confused).
-   **📊 Analytics Dashboard**:
    -   **Live Trend Chart**: Visualizes class focus levels over time.
    -   **Demographics**: Real-time Gender distribution stats.
    -   **Emotion Breakdown**: Pie charts showing the prevailing mood of the class.
-   **🔒 Privacy-First Design**: Video streams are processed in memory. No facial images are saved to disk.
-   **📝 Session Logging**: Automatically saves session statistics to a local SQLite database for historical review.
-   **📈 Validation Tools**: Built-in scripts to generate confusion matrices and accuracy reports for the underlying AI models.

---

## Overall Architecture

Teacher Station Pro follows a **Modular Layered Architecture**, separating the UI, Business Logic, and Data processing.

```mermaid
graph TD
    subgraph "Presentation Layer"
        UI[CustomTkinter UI]
        Dashboard[Live Dashboard]
    end

    subgraph "Service Layer"
        AppCore[App Core Logic]
        Capture[Capture Service]
        FaceService[Face Service]
    end

    subgraph "AI & Data Layer"
        FaceDet[Haar Cascade]
        EmoNet[Emotion Model (Mini_XCEPTION)]
        GenNet[Gender Model (Mini_XCEPTION)]
        DB[(SQLite Logs)]
    end

    UI --> AppCore
    AppCore --> Capture
    Capture --> FaceService
    FaceService --> FaceDet
    FaceService --> EmoNet
    FaceService --> GenNet
    AppCore --> DB
    AppCore --> Dashboard
```

### Technology Stack
-   **Language**: Python 3.9+
-   **GUI**: CustomTkinter (Modern, Dark-mode supported UI)
-   **Computer Vision**: OpenCV (Image processing & Face Detection)
-   **Deep Learning**: TensorFlow/Keras (Mini_XCEPTION architecture for Emotion & Gender)
-   **Data**: SQLite (Session persistence), Pandas/Matplotlib (Reporting)

---

## Installation

### Prerequisites
-   Windows 10/11 (Recommended)
-   Python 3.9 or higher
-   Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/teacher-station-pro.git
cd teacher-station-pro
```

### 2. Set Up Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows:
.
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r REQUIREMENTS.txt
```

### 4. Model Setup (Crucial Step)
The deep learning models are stored in the `trained_models` directory. Ensure the following files exist (extract them if they are in a zip file):
-   `trained_models/emotion_models/fer2013_mini_XCEPTION.107-0.66.hdf5`
-   `trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5`

> **Note:** If these files are missing, the application will default to dummy predictions or fail to initialize the AI service.

---

## Running the Application

### Method 1: Via Python CLI
Ensure your virtual environment is activated:
```bash
python main.py
```

### Method 2: Via VBScript (Windows)
Double-click the `Launch_App.vbs` file in the root directory. This will automatically handle the environment activation (if configured) and launch the app without a persistent console window.

---

## Folder Structure

```plaintext
D:\SE\
├── configs/                 # Configuration files (if any)
├── datasets/                # Validation datasets (for report generation)
├── images/                  # Static assets for UI or documentation
├── reports/                 # Generated session reports and logs
├── src/
│   ├── data/                # Database interactions and repositories
│   ├── models/              # Model architecture definitions (PyTorch/Keras)
│   ├── services/            # Core business logic (Face processing, Video capture)
│   └── ui/                  # UI components (Windows, Charts, Widgets)
├── trained_models/          # Pre-trained AI model weights (.hdf5, .pth)
├── Launch_App.vbs           # Windows quick-launcher
├── main.py                  # Application Entry Point
└── REQUIREMENTS.txt         # Python dependencies
```

---

## Environment Configuration

This project is designed to be **Zero-Config**. It uses relative paths to locate resources.
-   **Database**: Automatically creates `app_data.db` (or similar) in the local folder.
-   **Camera**: Automatically detects the default webcam (Index 0). External cameras can be configured in the source code (`capture_service.py`).

---

## Roadmap

- [x] **Phase 1**: Core AI Model Integration (Emotion & Gender).
- [x] **Phase 2**: Real-time Video Pipeline Construction.
- [x] **Phase 3**: UI/UX Design with CustomTkinter & Dashboarding.
- [ ] **Phase 4**: Advanced Reporting (PDF Export, Email alerts).
- [ ] **Phase 5**: Multi-camera support for large lecture halls.
- [ ] **Phase 6**: Student individual tracking (with opt-in Identity Recognition).

---

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
  Built with ❤️ by the Teacher Station Pro Team
</p>
