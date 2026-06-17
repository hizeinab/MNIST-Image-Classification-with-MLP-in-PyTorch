# MNIST-Image-Classification-with-MLP-in-PyTorch
This notebook demonstrates how to build and train a simple Multilayer Perceptron (MLP) neural network using PyTorch to classify handwritten digits from the MNIST dataset. It covers essential steps from data loading and preprocessing to model definition, training, and evaluation.

Table of Contents
1. Setting up the Environment and Loading Data
1.1 Imports
1.2 Device Management
1.2 Data Loading & Transformations
1.3 Exploring the Dataset Structure
1.4 Preparing Data with DataLoaders
2. Defining the Model Architecture
2.1 Multilayer Perceptron (MLP) Class
2.2 Model Instantiation and Parameter Count
3. Loss Function and Optimizer
3.1 Loss Function (nn.CrossEntropyLoss)
3.2 Optimizer (torch.optim.Adam)
4. Implementing the Training Loop
4.1 Epoch and Batch Processing
4.2 Tracking Metrics
5. Performance Evaluation
5.1 Visualizing Training Progress
5.2 Interpretation of Loss and Accuracy Curves
5.3 Final Test Set Evaluation
5.4 Confusion Matrix Analysis
6. Key PyTorch Training Concepts (Prerequisites)
6.1 Reshaping Inputs for the MLP (.view())
6.2 Understanding Model Outputs (.data vs. .detach() vs. torch.no_grad())
6.3 Converting Model Probabilities to Predicted Labels
6.4 Extracting Scalar Values (.item())
6.5 General Python Formatting (F-strings)
