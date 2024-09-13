Overview

This project showcases a simple neural network built from scratch using NumPy for matrix operations and Keras to load the MNIST dataset. It demonstrates my ability to develop fundamental machine learning architectures, optimize neural networks, and apply core concepts such as backpropagation, gradient descent, and early stopping. The model achieves competitive accuracy on the MNIST dataset, which consists of handwritten digits, and is optimized with the Adam optimizer.	  

Key Features  
-Custom-built Neural Network Layers: Implemented dense (fully connected) layers with backpropagation.  
-Activation Functions:		ReLU for hidden layers to introduce non-linearity.  
-Softmax for the output layer to handle multi-class classification.  
-Batch Normalization: Used to improve learning stability and performance.  
-Dropout: Incorporated dropout layers to prevent overfitting.  
-Categorical Crossentropy Loss: Standard loss function for multi-class classification problems.  
-Adam Optimizer: A well-known optimization technique with adaptive learning rates.  
-Early Stopping Mechanism: To avoid overfitting by monitoring validation accuracy.  
  
Project Structure  
Simple_Neural.py: The core file that contains the implementation of the entire neural network. This includes layer definitions, forward and backward passes, as well as the training loop.
saved_data/: Directory where trained model weights and biases are saved.
  
Implementation Details  
Data Preparation:  
Loaded the MNIST dataset (60,000 training images, 10,000 test images).  
Reshaped and normalized the image data for efficient model training.  
Split the training data into a training and validation set.  
  
Model Architecture:  
Input Layer: 784 inputs (28x28 images flattened).  
Hidden Layers:Three fully connected layers with ReLU activation and batch normalization.  
Dropout layers with a rate of 0.1 for regularization.  
Output Layer: 10 neurons with Softmax activation to classify digits (0-9).  
  
Training:    
The model is trained over 20 epochs.  
Monitored loss and accuracy on both training and validation sets.  
Adjusted the learning rate dynamically based on validation performance.  
Saving the Model: The model's weights and biases are saved to the saved_data folder after training for easy reuse or fine-tuning.  
  
Results  
The model achieves competitive accuracy (~88%) on the validation set, demonstrating a solid understanding of neural network construction and optimization.  

Why This Project Matters  
By building a custom neural network without relying on high-level libraries like TensorFlow or PyTorch, I show that I can work at the foundational level of deep learning, making me a valuable asset in any machine learning role.  

Implementing machine learning algorithms from scratch.
Understanding the mechanics of neural networks (e.g., forward/backward propagation).
Applying advanced optimization techniques like Adam.
Using real-world datasets to develop robust models.
By building a custom neural network without relying on high-level libraries like TensorFlow or PyTorch, I show that I can work at the foundational level of deep learning, making me a valuable asset in any machine learning role.
