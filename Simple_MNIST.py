import numpy as np
import nnfs
# from nnfs.datasets import spiral_data
from keras.datasets import mnist

# nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs  # Зберігаємо вхідні дані
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Градієнти зворотного поширення
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = np.multiply(dvalues, np.int64(self.output > 0))


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # Якщо мітки у форматі "sparse", перетворити їх у one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]

        # Обчислення градієнту
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


def to_one_hot(labels, num_classes):
    one_hot_labels = np.eye(num_classes)[labels]
    return one_hot_labels

# Завантаження набору даних MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормалізація і решейпінг даних
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# Перетворення міток у one-hot вектори
train_labels_one_hot = to_one_hot(train_labels, 10)  # 10 класів для MNIST
test_labels_one_hot = to_one_hot(test_labels, 10)

X = train_images
Y = train_labels_one_hot

# Ініціалізація шарів та функцій активації
dense1 = Layer_Dense(784, 128)  # 784 вхідних особливості, 128 нейронів
activation1 = Activation_ReLU()

dense2 = Layer_Dense(128, 10)   # 128 вхідних особливостей, 10 нейронів (класів MNIST)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# Ініціалізація оптимізатора
optimizer = Optimizer_SGD(learning_rate=1.0)

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, Y)

for epoch in range(100):  # Кількість епох
    # Пряме поширення
    dense1.forward(train_images)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Втрати
    loss = loss_function.calculate(activation2.output, train_labels)

    # Зворотне поширення (backward)
    loss_function.backward(activation2.output, Y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Оновлення ваг і зміщень
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

def predict(model, X):
    # Пряме поширення через всі шари
    for layer in model:
        layer.forward(X)
        X = layer.output
    # Повернення індексу найбільшого значення в кожному виході
    return np.argmax(X, axis=1)

def accuracy(y_pred, y_true):
    # Обчислення точності як відсоток правильних передбачень
    return np.mean(y_pred == y_true)

# Формування тестового набору даних
X_test, y_test = test_images, test_labels

# Роблення передбачень на тестовому наборі даних
y_pred = predict([dense1, activation1, dense2, activation2], X_test)

# Переведення міток y_test у числовий формат, якщо вони у форматі one-hot
y_test = np.argmax(y_test, axis=0)

# Обчислення точності
test_accuracy = accuracy(y_pred, y_test)
print("Точність на тестовому наборі даних:", test_accuracy)