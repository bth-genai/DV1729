import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Train and visualize a neural network for logic gates.")
parser.add_argument("gate", choices=["AND", "OR", "XOR"], help="The logic gate to learn (AND, OR, XOR)")
args = parser.parse_args()

# 1. SETUP DATA (AND Gate)
# The four conbinations
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)

if args.gate == "AND":
    y = torch.tensor([[0.], [0.], [0.], [1.]], dtype=torch.float32)
    # Linear model is sufficient
    model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    epochs = 3000
elif args.gate == "OR":
    y = torch.tensor([[0.], [1.], [1.], [1.]], dtype=torch.float32)
    # Linear model is sufficient
    model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    epochs = 3000
elif args.gate == "XOR":
    y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)
    # Non-linear model required (Hidden Layer)
    model = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid())
    epochs = 10000

print(f"Dataset Created!\nInput Shape: {X.shape}\nTarget Shape: {y.shape}\n")

# 2. TRAIN MODEL (same as lab)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCELoss()

print(f"Training the model for {args.gate}...")
print(f"Initial prediction for [1, 1] (before training): {model(torch.tensor([1.0, 1.0])).item():.4f}")

for epoch in range(epochs):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

print("\n--- Final Results ---")
with torch.no_grad():
    test_input = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    prediction = model(test_input)
    print(f"Testing [1, 0] -> Model says: {prediction[0].item():.4f} (Rounded: {round(prediction[0].item())})")
    print(f"Testing [1, 1] -> Model says: {prediction[1].item():.4f} (Rounded: {round(prediction[1].item())})")

if args.gate == "XOR":
    print(f"\nLearned Hidden Weights:\n{model[0].weight.data}")
    print(f"Learned Output Weights:\n{model[2].weight.data}")
else:
    w = model[0].weight.data[0]
    b = model[0].bias.data[0]
    print(f"\nLearned Weights: {w}")
    print(f"Learned Bias: {b}")
    
    # Calculate standard line equation: x2 = m*x1 + c
    # Derived from: w1*x1 + w2*x2 + b = 0  ->  x2 = (-w1/w2)*x1 + (-b/w2)
    w1, w2 = w[0].item(), w[1].item()
    bias_val = b.item()
    
    if abs(w2) > 0.0001: # Avoid division by zero
        slope = -w1 / w2
        intercept = -bias_val / w2
        print(f"Line Equation: x2 = {slope:.2f} * x1 + {intercept:.2f}")
        print(f"  -> Slope (m): {slope:.2f} (determined by weights)")
        print(f"  -> Intercept (c): {intercept:.2f} (determined by bias and weights)")

# 3. Visualize
def plot_decision_boundary(model, X, y):
    # Convert tensors to numpy arrays to avoid Matplotlib errors
    X = X.numpy()
    y = y.numpy()

    # Create a square flat space (-0.5 to 1.5)
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Check what the model outputs for every point in the space
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    with torch.no_grad():
        pred = model(grid_tensor).reshape(xx.shape)

        # If probability > 0.5, set to 1. Otherwise 0.
        pred_solid = (pred > 0.5).float()
        
        # Convert to numpy for plotting
        pred = pred.numpy()
        pred_solid = pred_solid.numpy()
    
    # Create color map
    # Background: light red (for 0) and light blue (for 1)
    cmap_background = ListedColormap(['#FFDDDD', '#DDDDFF'])
    # Dots: dark red (for 0) och dark blue (for 1)
    cmap_dots = ListedColormap(['#AA0000', '#0000AA'])

    # Draw the background
    plt.pcolormesh(xx, yy, pred_solid, cmap=cmap_background, shading='auto')
    
    # Draw a black line for the models linear limit (where the model is 50% certain)
    plt.contour(xx, yy, pred, levels=[0.5], colors='black', linewidths=3)

    # Draw the data points
    # c=y.flatten() flatten limits to (0 or 1)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=250, 
                cmap=cmap_dots, edgecolors='white', linewidth=2)

    # Set title and limit the plot
    plt.title(f"Neural Network Decision Boundary ({args.gate} Gate)")
    plt.xlabel("Input 1 (Logic switch A)", fontsize=12)
    plt.ylabel("Input 2 (Logic switch B)", fontsize=12)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid(False)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    
    # Draw the actual axes at 0 so we can see the origin
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
    plt.show(block=True)

print("Plotting decision boundary...")
plot_decision_boundary(model, X, y)