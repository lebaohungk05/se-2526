import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '../trained_models/fer2013_mini_XCEPTION.119-0.65.hdf5') 
LOG_PATH = os.path.join(SCRIPT_DIR, '../trained_models/emotion_models/fer2013_emotion_training.log')
TEST_DATA_DIR = os.path.join(SCRIPT_DIR, '../datasets/test')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../report_images')
IMG_SIZE = (48, 48)
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("--- BAT DAU TAO BAO CAO ---")

# 1. VE BIEU DO TRAINING HISTORY
print("\n[1/3] Dang ve bieu do Training History...")
if os.path.exists(LOG_PATH):
    data = pd.read_csv(LOG_PATH)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data['epoch'], data['accuracy'], label='Train Accuracy')
    plt.plot(data['epoch'], data['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(data['epoch'], data['loss'], label='Train Loss')
    plt.plot(data['epoch'], data['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_final.png'))
    print("Done. Saved to report_images/training_history_final.png")

# 2. LOAD MODEL VA DU DOAN
print("\n[2/3] Dang load model va du doan tren tap Test...")
model = load_model(MODEL_PATH, compile=False)

X_test = []
y_test = []

# Load data thu cong de dam bao preprocess khop 100%
for emotion_idx, emotion_name in enumerate(CLASSES):
    folder = os.path.join(TEST_DATA_DIR, emotion_name)
    if not os.path.exists(folder): continue
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype('float32') / 255.0
            img = (img - 0.5) * 2.0
            X_test.append(np.expand_dims(img, -1))
            y_test.append(emotion_idx)

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred_prob = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)

# 3. CONFUSION MATRIX
print("\n[3/3] Dang ve Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

report = classification_report(y_test, y_pred, target_names=CLASSES)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

print("\n--- HOAN TAT ---")
print(report)
