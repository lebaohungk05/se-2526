import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import pandas as pd
import os
import time
import cv2
from PIL import Image

# --- Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
INPUT_SIZE = 112  # Resize to 112x112 for better feature extraction
VALIDATION_SPLIT = .2
NUM_CLASSES = 7
PATIENCE = 10
BASE_PATH = '../trained_models/emotion_models/'
DATASET_NAME = 'fer2013'
DATASET_PATH = '../datasets'

# --- Custom Dataset Class ---
class FER2013Dataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, val_split=0.2):
        """
        Args:
            root_dir (str): Path to datasets directory containing 'train' and 'test' folders.
            mode (str): 'train' or 'val'.
            transform (callable, optional): Transform to be applied on a sample.
            val_split (float): Fraction of training data to use for validation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Load all image paths and labels from the 'train' folder (we split this for train/val)
        # Note: The original dataset zip structure has 'train' and 'test'. 
        # Usually 'test' is held out completely. We split 'train' into 'train' and 'val'.
        train_dir = os.path.join(root_dir, 'train')
        
        self.samples = []
        self.class_to_idx = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
        
        # Collect all data first
        all_data = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_data.append((os.path.join(class_dir, img_name), class_idx))
        
        # Shuffle deterministically
        np.random.seed(42)
        np.random.shuffle(all_data)
        
        # Split
        split_idx = int(len(all_data) * (1 - val_split))
        if self.mode == 'train':
            self.samples = all_data[:split_idx]
        else: # val
            self.samples = all_data[split_idx:]
            
        print(f"[{mode.upper()}] Loaded {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image using PIL (better integration with torchvision)
        # FER2013 is grayscale, convert to RGB for MobileNetV2
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Main Script ---
if __name__ == "__main__":
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    # 2. Data Transforms
    # Standard ImageNet normalization
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Helpful even for grayscale->RGB
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 3. Load Data
    print("Initializing Datasets...")
    train_dataset = FER2013Dataset(DATASET_PATH, mode='train', transform=train_transform, val_split=VALIDATION_SPLIT)
    val_dataset = FER2013Dataset(DATASET_PATH, mode='val', transform=val_transform, val_split=VALIDATION_SPLIT)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Setup (Transfer Learning)
    print("Downloading/Loading Pretrained MobileNetV2...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    
    # Freeze all layers first (Optional: can also fine-tune all immediately, but freezing head first is safer)
    # For this task, fine-tuning all layers usually works better given the domain shift (Objects -> Faces)
    # So we will NOT freeze feature extractor, but we use a lower learning rate.
    
    # Replace Classifier Head
    # MobileNetV2 classifier is a Sequential block: (0): Dropout, (1): Linear
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    model = model.to(device)

    # 5. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Lower LR for pretrained weights, higher LR for new head could be better, but simpler is fine.
    optimizer = Adam(model.parameters(), lr=1e-4) 
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)

    # 6. Training Loop
    log_file_path = os.path.join(BASE_PATH, f'{DATASET_NAME}_transfer_learning.log')
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr'])
    best_val_acc = 0.0
    
    print(f"Starting Transfer Learning for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Scheduler
        scheduler.step(avg_val_loss)
        
        duration = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Time: {duration:.1f}s | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr}")
        
        # Logging
        log_entry = pd.DataFrame([{'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'train_acc': train_acc, 'val_acc': val_acc, 'lr': current_lr}])
        log_df = pd.concat([log_df, log_entry], ignore_index=True)
        log_df.to_csv(log_file_path, index=False)

        # Save Best Model (Based on Accuracy)
        if val_acc > best_val_acc:
            print(f"Found better model! ({best_val_acc:.2f}% -> {val_acc:.2f}%) Saving...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(BASE_PATH, f'{DATASET_NAME}_mobilenet_v2_transfer_best.pth'))

    print("Training Complete.")
