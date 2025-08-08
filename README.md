Machine learning course 
Assignment 1
Digit Recognition Using PyTorch


What is Digit Recognition Using PyTorch?
Digit recognition using PyTorch is a machine learning project where we use a deep learning model (neural network) built with the PyTorch framework to automatically identify handwritten digits (0 through 9) from image data — typically using the MNIST dataset.

The Goal:
To build a model that takes a 28x28 grayscale image of a handwritten digit as input and predicts the correct digit (0–9) as output.

Dataset Used: MNIST
60,000 training images
10,000 test images
Each image is 28x28 pixels
Each image contains a single digit (0–9)

Why Use PyTorch?
PyTorch is a powerful, open-source deep learning library that allows:
Easy construction of neural networks (torch.nn)
Efficient training using automatic backpropagation
GPU acceleration (using CUDA)
Flexible model evaluation and deployment

Steps Involved in Digit Recognition:
1. Load and Transform Data
Use torchvision.datasets.MNIST to download the dataset.
Apply transformations (ToTensor, Normalize) to convert images into model-ready tensors.
2. Define the Neural Network
A simple fully connected neural network is created.
Layers:
Input layer: 784 neurons (flattened 28x28 image)
Hidden layer: 128 neurons + ReLU activation
Output layer: 10 neurons (one for each digit)
3. Train the Model
Use a loss function (CrossEntropyLoss) to measure how wrong the predictions are.
Use an optimizer (Adam) to update the model's weights.
Train for several epochs on batches of data.
4. Evaluate the Model
After training, test the model on unseen data.
Measure accuracy to see how well the model predicts digits.
