import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import numpy as np
import os
import time
import csv
import cv2
from torch.utils.tensorboard import SummaryWriter 

from models.mobilenet_v2_pytorch import mobilenet_v2_pytorch
from utils.preprocessor import preprocess_input

# --- Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 10000 
INPUT_SHAPE = (96, 96, 1) # Increased to 96x96 to boost Accuracy > 66%
NUM_CLASSES = 7
EARLY_STOPPING_PATIENCE = 25 # Tăng một chút để model có cơ hội hội tụ sâu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, '../trained_models/emotion_models/')
DATASET_NAME = 'fer2013'
DATASET_PATH = os.path.join(SCRIPT_DIR, '../datasets')
TENSORBOARD_LOG_DIR = os.path.join(BASE_PATH, 'tensorboard_logs')
CSV_LOG_PATH = os.path.join(BASE_PATH, f'{DATASET_NAME}_training_log.csv')

ALPHA = 1.0
DROPOUT_RATE = 0.5

def load_data_from_dir(directory, image_size=(64, 64)):
    """Hàm load dữ liệu từ cấu trúc thư mục train/test"""
    class_to_arg = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    faces = []
    emotions = []
    for emotion_name, emotion_idx in class_to_arg.items():
        folder = os.path.join(directory, emotion_name)
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    faces.append(img.astype('float32'))
                    emotions.append(emotion_idx)
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1) # Thêm channel axis (H, W, 1)
    return faces, np.array(emotions)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(BASE_PATH): os.makedirs(BASE_PATH)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    # 1. Load Data (Load riêng Train và Test)
    print("Loading datasets...")
    train_faces, train_emotions = load_data_from_dir(os.path.join(DATASET_PATH, 'train'), INPUT_SHAPE[:2])
    test_faces, test_emotions = load_data_from_dir(os.path.join(DATASET_PATH, 'test'), INPUT_SHAPE[:2])
    
    # Preprocess
    train_faces = preprocess_input(train_faces)
    test_faces = preprocess_input(test_faces)
    
    # Transpose to (N, C, H, W)
    train_faces = np.transpose(train_faces, (0, 3, 1, 2))
    test_faces = np.transpose(test_faces, (0, 3, 1, 2))

    # To Tensors
    train_x = torch.from_numpy(train_faces).float()
    train_y = torch.from_numpy(train_emotions).long()
    test_x = torch.from_numpy(test_faces).float()
    test_y = torch.from_numpy(test_emotions).long()

    # 2. Augmentation (Thêm ColorJitter để xử lý ánh sáng)
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Giúp model bền bỉ với ánh sáng
    ])

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model & Optimizer
    model = mobilenet_v2_pytorch(input_shape=(1, INPUT_SHAPE[0], INPUT_SHAPE[1]), 
                                num_classes=NUM_CLASSES, alpha=ALPHA, dropout=DROPOUT_RATE)
    model.to(device)

    # Sử dụng Label Smoothing để tăng khả năng tổng quát hóa
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    # Resume logic
    best_model_path = os.path.join(BASE_PATH, f'{DATASET_NAME}_mobilenet_v2_pytorch_best.pth')
    checkpoint_path = os.path.join(BASE_PATH, f'{DATASET_NAME}_mobilenet_v2_pytorch_latest.pth')
    start_epoch, best_val_loss, early_stop_counter = 0, float('inf'), 0

    if os.path.exists(checkpoint_path):
        print("Resuming from latest checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        early_stop_counter = ckpt.get('early_stop_counter', 0)

    # 4. Training Loop
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'lr'])

    print(f"Training started. Target Accuracy: >66%")

    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = train_transform(batch_x)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (pred == batch_y).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_acc = 100 * val_correct / val_total
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        scheduler.step(avg_val_loss)
        
        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr}")
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch+1)
        
        with open(CSV_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, avg_val_loss, val_acc, lr])

        # Save Checkpoints
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  --> New Best Model Saved!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered."); break

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss, 'early_stop_counter': early_stop_counter
        }, checkpoint_path)

    writer.close()
    print("Training Complete.")