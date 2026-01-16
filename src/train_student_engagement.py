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
from PIL import Image

# --- Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
INPUT_SIZE = 224  # Resize to 224x224 (Standard for MobileNetV2)
VALIDATION_SPLIT = .2
NUM_CLASSES = 7
PATIENCE = 10

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, '../trained_models/emotion_models/')
DATASET_NAME = 'fer2013'
DATASET_PATH = os.path.join(SCRIPT_DIR, '../datasets')

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
        
        # Determine strict path to train directory
        train_dir = os.path.join(root_dir, 'train')
        if not os.path.exists(train_dir):
             # Fallback or check structure
             train_dir = os.path.join(root_dir, 'fer2013', 'train')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Could not find 'train' directory in {root_dir}")

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
            
        print(f"[{mode.upper()}] Loaded {len(self.samples)} images from {train_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image using PIL (better integration with torchvision)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image or handle error
            image = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE))

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
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 3. Load Data
    print(f"Initializing Datasets from {DATASET_PATH}...")
    try:
        train_dataset = FER2013Dataset(DATASET_PATH, mode='train', transform=train_transform, val_split=VALIDATION_SPLIT)
        val_dataset = FER2013Dataset(DATASET_PATH, mode='val', transform=val_transform, val_split=VALIDATION_SPLIT)
    except Exception as e:
        print(f"Error initializing datasets: {e}")
        exit(1)

    # Windows often requires num_workers=0 or guarded entry point
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # --- Calculate Class Weights ---
    print("Calculating class weights...")
    class_counts = np.zeros(NUM_CLASSES)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    # Avoid division by zero
    class_counts[class_counts == 0] = 1
    
    class_weights = len(train_dataset) / (NUM_CLASSES * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Class Counts: {class_counts}")
    print(f"Class Weights: {class_weights}")

    # 4. Model Setup (Transfer Learning)
    print("Downloading/Loading Pretrained MobileNetV2...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    
    # Replace Classifier Head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    model = model.to(device)
    
    # 5. Load Checkpoint if exists (Resume Training)
    checkpoint_path = os.path.join(BASE_PATH, f'{DATASET_NAME}_student_engagement_best.pth')
    start_epoch = 0
    best_val_acc = 0.0
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}. Loading...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Checkpoint loaded successfully.")
            # Ideally we would also load optimizer state, epoch, etc. if we saved them.
            # Since we only saved state_dict, we start fresh with optimizer but with pre-trained weights.
            # We can verify accuracy on validation set to set best_val_acc
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            best_val_acc = 100 * correct / total
            print(f"Resuming with validation accuracy: {best_val_acc:.2f}%")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from ImageNet weights.")

    # 6. Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    optimizer = Adam(model.parameters(), lr=1e-4) 
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=True)

    # 7. Training Loop
    log_file_path = os.path.join(BASE_PATH, f'{DATASET_NAME}_student_engagement_training.log')
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr'])
    best_val_acc = 0.0
    
    print(f"Starting Training (Student Engagement Model) for {NUM_EPOCHS} epochs...")
    
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
            torch.save(model.state_dict(), os.path.join(BASE_PATH, f'{DATASET_NAME}_student_engagement_best.pth'))

    print("Training Complete.")