import matplotlib
matplotlib.use('Agg')  # Chi dinh backend, phai dat truoc khi import pyplot
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Đường dẫn đến file log (tương đối từ thư mục src)
log_file = '../trained_models/emotion_models/fer2013_emotion_training.log'
output_file = 'training_history.png'

print(f"--- Bat dau ve bieu do ---")
print(f"Kiem tra file log tai: {os.path.abspath(log_file)}")

# Kiem tra su ton tai cua file log
if not os.path.exists(log_file):
    print(f"\n[LOI] File log khong ton tai: {log_file}")
    print("Ban can chay training de file nay duoc tao ra.")
    sys.exit(1)

# Kiem tra file log co du lieu khong
if os.path.getsize(log_file) == 0:
    print(f"\n[LOI] File log dang rong. Hay doi epoch dau tien hoan thanh.")
    sys.exit(1)

print("File log da ton tai va co du du lieu.")

try:
    # Doc du lieu tu file CSV
    data = pd.read_csv(log_file)
    print("Doc file log thanh cong.")
    
    # Kiem tra xem co cot du lieu can thiet khong
    required_columns = ['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss']
    if not all(col in data.columns for col in required_columns):
        print(f"\n[LOI] File log thieu cot du lieu. Cac cot hien co: {list(data.columns)}")
        sys.exit(1)

    # Thiet lap bieu do
    plt.figure(figsize=(14, 6))
    plt.suptitle('Training History Analysis', fontsize=16)

    # Bieu do Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(data['epoch'], data['accuracy'], label='Train Accuracy', marker='.')
    plt.plot(data['epoch'], data['val_accuracy'], label='Val Accuracy', marker='.')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # Bieu do Loss
    plt.subplot(1, 2, 2)
    plt.plot(data['epoch'], data['loss'], label='Train Loss', marker='.')
    plt.plot(data['epoch'], data['val_loss'], label='Val Loss', marker='.')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Luu bieu do
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    
    print(f"\n[THANH CONG] Da ve va luu bieu do vao file:")
    print(f"{os.path.abspath(output_file)}")

except Exception as e:
    print(f"\n[LOI] Da xay ra loi khong xac dinh khi ve bieu do: {e}")
    sys.exit(1)
