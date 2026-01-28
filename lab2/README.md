# Lab 2
Using pytorch

main.py is a slightly modified basic example from the Pytorch Quickstart (https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html). It is using the MNIST Fashion dataset (https://www.kaggle.com/code/texasdave/image-classification-tutorial-with-mnist-fashion)

test_mnist.py is a script to test an image against the trained model. Images must be converted to 28x28 pixels. The dataset used to train the model consists of grayscale images of clothing on black background, so the image to classify should be in the same style.

First, we need to train the model:
```
uv run tutorial_example.py
```
Then we can try to classify an image using the model:
```
uv run test_mnist.py
```

Feel free to continue learning about the basics of Pytorch: https://docs.pytorch.org/tutorials/beginner/basics/intro.html, it will give you a great insight in how neural networks and AI work.
