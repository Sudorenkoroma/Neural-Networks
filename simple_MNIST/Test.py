import numpy as np
import cv2


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs  # Зберігаємо вхідні дані
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


dense1 = Layer_Dense(784, 512)
dense2 = Layer_Dense(512, 256)
dense3 = Layer_Dense(256, 128)
dense4 = Layer_Dense(128, 10)

activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_ReLU()
activation4 = Activation_Softmax()

import os
folder_path = "saved_data"
os.makedirs(folder_path, exist_ok=True)

dense1.weights = np.load(os.path.join(folder_path, 'dense1_weights.npy'))
dense1.biases = np.load(os.path.join(folder_path,'dense1_biases.npy'))
dense2.weights = np.load(os.path.join(folder_path,'dense2_weights.npy'))
dense2.biases = np.load(os.path.join(folder_path,'dense2_biases.npy'))
dense3.weights = np.load(os.path.join(folder_path,'dense3_weights.npy'))
dense3.biases = np.load(os.path.join(folder_path,'dense3_biases.npy'))
dense4.weights = np.load(os.path.join(folder_path,'dense4_weights.npy'))
dense4.biases = np.load(os.path.join(folder_path,'dense4_biases.npy'))

def predict(model, X):
    # Пряме поширення через всі шари
    for layer in model:
        layer.forward(X)
        X = layer.output
    # Повернення індексу найбільшого значення в кожному виході
    return np.argmax(X, axis=1)

img = cv2.imread('test_img.png', cv2.IMREAD_GRAYSCALE)

# Перевірка на наявність зображення
if img is not None:
    # Зміна розміру зображення до 28x28
    img = cv2.resize(img, (28, 28))
    # Інвертування зображення: в MNIST білі цифри на чорному фоні
    img = 255 - img
    # Нормалізація піксельних значень
    img = img.astype('float32') / 255
    # Решейпінг зображення для мережі (1, 784)
    img = img.reshape(1, 784)
    # Використання моделі для передбачення цифри
    prediction = predict([dense1, activation1, dense2, activation2, dense3, activation3, dense4, activation4], img)

    # Виведення передбачення
    print("Цифра яку попросив написать Сул:", prediction[0])
else:
    print("Зображення не знайдено.")