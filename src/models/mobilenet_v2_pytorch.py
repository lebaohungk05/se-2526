import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

def mobilenet_v2_pytorch(input_shape=(1, 64, 64), num_classes=7, alpha=1.0, dropout=0.5):
    """
    Constructs a PyTorch MobileNetV2 model adapted for emotion classification.

    Args:
        input_shape: tuple, e.g. (1, 64, 64) for (channels, height, width)
        num_classes: int, number of emotion classes
        alpha: float, width multiplier for MobileNetV2.
        dropout: float, dropout rate for the classifier head.
    """
    # Load the standard MobileNetV2 model from torchvision
    # We don't use pre-trained weights from ImageNet because we're training from scratch
    # on a different domain (grayscale facial expressions).
    model = mobilenet_v2(weights=None, width_mult=alpha)

    # The original Keras model handled 1-channel input implicitly.
    # In PyTorch, we need to adapt the first convolutional layer.
    # Original first layer: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # We'll change it to accept 1 input channel instead of 3.
    original_first_layer = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=original_first_layer.out_channels,
        kernel_size=original_first_layer.kernel_size,
        stride=original_first_layer.stride,
        padding=original_first_layer.padding,
        bias=False
    )

    # Replace the classifier head. 
    # Original classifier: Sequential(Dropout(p=0.2), Linear(in_features=1280, out_features=1000, bias=True))
    # We rebuild it with our desired dropout and num_classes.
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_ftrs, num_classes),
    )

    return model

if __name__ == '__main__':
    # Example of how to create the model
    # PyTorch uses Channels-Height-Width (CHW) format
    input_c, input_h, input_w = 1, 64, 64
    
    # Create the model
    model = mobilenet_v2_pytorch(input_shape=(input_c, input_h, input_w), num_classes=7, alpha=1.0)
    
    # Print model summary (requires torchinfo or similar, but we can just print the structure)
    print(model)
    
    # Example forward pass
    # Create a dummy input tensor
    dummy_input = torch.randn(32, input_c, input_h, input_w) # batch_size=32
    print(f"\nInput shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (32, 7)
    print("Model created and forward pass successful!")
