import os
from matplotlib import pyplot as plt
from PIL import Image

import torch
from torch import nn
from torchvision.transforms import ToTensor

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def predict_custom_image(model, image_path, device, classes):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Load image, convert to grayscale, resize to 28x28
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    
    # Convert to tensor, add batch dimension (N, C, H, W)
    x = ToTensor()(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred = model(x)
        predicted = classes[pred[0].argmax(0)]
        print(f'Custom Image "{image_path}" Predicted: "{predicted}"')
        plt.imshow(img, cmap='gray')
        plt.show()

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model = NeuralNetwork().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Model loaded.")
    else:
        print("Model file not found. Please run tutorial_example.py first to train the model.")

    file_path = os.path.join(os.path.dirname(__file__), "tshirt1.png")
    predict_custom_image(model, file_path, device, classes)

if __name__ == "__main__":
    main()
