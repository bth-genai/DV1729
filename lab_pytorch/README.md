# Lab Pytorch
**Topic:** Computer Vision & Tensors  
**Prerequisites:** Python installed, Basic terminal usage.

## 🎯 Objective
In this lab, you will leave the "low-code" world and write code using **PyTorch**, the industry-standard machine learning framework.

You will:
1.  **Part 1:** Train a Neural Network to recognize articles of clothing (FashionMNIST dataset).
2.  **Part 2:** Build a tiny neural network from scratch using Tensors to solve a logic puzzle.

---
## 👕 Part 1: Image Classification (FashionMNIST)
In this part, we use a classic dataset consisting of 60,000 grayscale images of clothes to train a model.

Read through tutorial_example.py. The script is a slightly modified basic example from the Pytorch Quickstart (https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html). It is using the MNIST Fashion dataset (https://www.kaggle.com/code/texasdave/image-classification-tutorial-with-mnist-fashion)

test_mnist.py is a script to test an image against the trained model. Images must be converted to 28x28 pixels. The dataset used to train the model consists of grayscale images of clothing on black background, so the image to classify should be in the same style.

First, we need to train the model:
``` bash
uv run tutorial_example.py
```
Then we can try to classify an image using the model:
``` bash
uv run test_mnist.py
```
---

## 🔧 Part 2: Building from Scratch (The "Hello World" of Tensors)

In the previous step, you used a ready-made big dataset. Now, you will do the opposite. You will build a dataset manually using **Tensors** (matrices) consisting of only 1s and 0s to understand how the learning works.

**The Task:** Teach a Neural Network to solve the **"Logical AND"** problem.
* Input `[0, 0]` → Output `0`
* Input `[0, 1]` → Output `0`
* Input `[1, 0]` → Output `0`
* Input `[1, 1]` → Output `1` (Only true if BOTH are true)

### Step 1: Create the Script
Create a new file named `tensor_lab.py`.

### Step 2: The Code
Copy the code below. Read the code and comments and try to explain exactly how Tensors and Gradients work.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# 1. CREATE THE DATASET (MANUAL TENSORS)
# ---------------------------------------------------------
# A Tensor is just a matrix that can be moved to GPU.
# We create a simple Truth Table for an "AND" gate.

# Inputs (X): 4 examples, each has 2 values (0 or 1)
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

# Targets (y): The correct answers.
# 0 = False, 1 = True. We shape it as 4 rows, 1 column.
y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0]
], dtype=torch.float32)

print(f"Dataset Created!\nInput Shape: {X.shape}\nTarget Shape: {y.shape}\n")


# ---------------------------------------------------------
# 2. DEFINE THE MODEL
# ---------------------------------------------------------
# Since this problem is simple, we actually only need ONE Neuron (Linear Layer).
# It will be "learned" the formula: y = weight1 * input1 + weight2 * input2 + bias

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer: 2 inputs -> 1 output
        self.layer = nn.Linear(2, 1)
        # Activation: Sigmoid limits the result between 0 and 1
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

model = TinyModel()

# ---------------------------------------------------------
# 3. THE TRAINING LOOP (The "Learning" part)
# ---------------------------------------------------------
# Loss Function: Measures how wrong the model is (BCELoss is efficient for binary 1/0)
loss_fn = nn.BCELoss()

# Optimizer: Adjusts the weights to reduce error (SGD = Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Starting Training...")
print(f"Initial prediction for [1, 1] (before training): {model(torch.tensor([1.0, 1.0])).item():.4f}")

epochs = 2000 # Run through the data 2000 times
for epoch in range(epochs):
    # A. Forward Pass (Make a guess)
    predictions = model(X)

    # B. Calculate Loss (How wrong were we?)
    loss = loss_fn(predictions, y)

    # C. Backward Pass (This is were we find how to tune the model!)
    # This calculates how much each weight contributed to the error.
    optimizer.zero_grad() # Reset old gradients
    loss.backward()       # Calculate new gradients

    # D. Step (Update weights)
    optimizer.step()      # Nudge weights in the correct direction

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# ---------------------------------------------------------
# 4. TEST THE RESULTS
# ---------------------------------------------------------
print("\n--- Final Results ---")
with torch.no_grad(): # We don't need to track gradients for testing
    test_input = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    prediction = model(test_input)
    
    # We round the output: anything above 0.5 becomes 1
    print(f"Testing [1, 0] -> Model says: {prediction[0].item():.4f} (Rounded: {round(prediction[0].item())})")
    print(f"Testing [1, 1] -> Model says: {prediction[1].item():.4f} (Rounded: {round(prediction[1].item())})")

# Let's inspect the weights directly!
weights = model.layer.weight.data
bias = model.layer.bias.data
print(f"\nLearned Weights: {weights}")
print(f"Learned Bias: {bias}")
```

### Step 3. Run the Experiment ###
```bash
uv run tensor_lab.py
```

## 🧠 Reflection
Look at the learned weights in the output. What weights did the model learn? Why do you think that is?

# Continue on your own
## Challenge: ##
Change the dataset y to represent an OR gate instead (Output is 1 if any input is 1). Rerun the training. Does it learn as fast or faster or slower?
If you instead try to "learn" an XOR gate, will it suffice with only 1 neuron?
## More resources ##
Feel free to continue learning about the basics of Pytorch: https://docs.pytorch.org/tutorials/beginner/basics/intro.html, it will give you a great insight in how neural networks and AI work.
