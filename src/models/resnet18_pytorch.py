import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def resnet18_pytorch(num_classes=7, pretrained=True):
    """
    ResNet-18 Custom:
    - Input 1 kênh (Grayscale)
    - Thêm Dropout vào giữa các Layer (Regularization mạnh)
    - FC Dropout cao (0.5)
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    # 1. Sửa Conv1 (3 kênh -> 1 kênh)
    original_conv = model.conv1
    new_conv = nn.Conv2d(1, original_conv.out_channels, 
                         kernel_size=original_conv.kernel_size, 
                         stride=original_conv.stride, 
                         padding=original_conv.padding, 
                         bias=False)
    
    if pretrained:
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)
    model.conv1 = new_conv

    # 2. INJECT DROPOUT vào giữa các tầng (Feature Extractor)
    # Giúp phá vỡ sự phụ thuộc giữa các khối features
    p_dropout = 0.1 # Nhẹ nhàng để không mất thông tin
    
    model.layer1 = nn.Sequential(model.layer1, nn.Dropout2d(p=p_dropout))
    model.layer2 = nn.Sequential(model.layer2, nn.Dropout2d(p=p_dropout))
    model.layer3 = nn.Sequential(model.layer3, nn.Dropout2d(p=p_dropout))
    model.layer4 = nn.Sequential(model.layer4, nn.Dropout2d(p=p_dropout))

    # 3. Fully Connected Layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model
