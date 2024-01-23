import numpy as np
from keras.datasets import mnist
import cv2


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
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


# Оптимізатор Adam
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradient
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights -= self.current_learning_rate * weight_momentums_corrected / (
                    np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / (
                    np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Layer_BatchNormalization:
    def __init__(self, n_inputs):
        self.gamma = np.ones((1, n_inputs))
        self.beta = np.zeros((1, n_inputs))
        self.moving_mean = np.zeros((1, n_inputs))
        self.moving_variance = np.ones((1, n_inputs))

    def forward(self, inputs, training=0):
        if training:
            mean = np.mean(inputs, axis=0, keepdims=True)
            variance = np.var(inputs, axis=0, keepdims=True)
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
            self.moving_variance = 0.9 * self.moving_variance + 0.1 * variance
        else:
            mean = self.moving_mean
            variance = self.moving_variance
        self.inputs_centered = inputs - mean
        self.stddev_inv = 1 / np.sqrt(variance + 1e-8)
        self.output = self.inputs_centered * self.stddev_inv * self.gamma + self.beta

    def backward(self, dvalues):
        samples = dvalues.shape[0]

        # Градієнти параметрів gamma та beta
        self.dgamma = np.sum(dvalues * self.inputs_centered * self.stddev_inv, axis=0, keepdims=True)
        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)

        # Градієнти по вхідним даним
        dinputs_centered = dvalues * self.gamma * self.stddev_inv
        dstddev_inv = np.sum(dvalues * self.gamma * self.inputs_centered, axis=0, keepdims=True)
        dvariance = dstddev_inv * -0.5 * (self.stddev_inv ** 3)
        dinputs_squared = dvariance * 2 / samples * np.ones_like(dvalues)
        dmean = np.sum(dinputs_centered + dinputs_squared * -2 * self.inputs_centered, axis=0, keepdims=True) * -1
        self.dinputs = dinputs_centered + dinputs_squared * 2 * self.inputs_centered / samples + dmean / samples

def to_one_hot(labels, num_classes):
    one_hot_labels = np.eye(num_classes)[labels]
    return one_hot_labels

# Завантаження та підготовка даних
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# Перетворення міток у one-hot вектори
train_labels = to_one_hot(train_labels, 10)
test_labels = to_one_hot(test_labels, 10)

# Розділення даних на тренувальні та валідаційні набори
validation_split = 0.1
validation_images = train_images[:int(validation_split * train_images.shape[0])]
validation_labels = train_labels[:int(validation_split * train_labels.shape[0])]
train_images = train_images[int(validation_split * train_images.shape[0]):]
train_labels = train_labels[int(validation_split * train_labels.shape[0]):]

# Оптимізація використання пам'яті
train_images = np.array(train_images)
train_labels = np.array(train_labels)
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)

# Ініціалізація моделі
dense1 = Layer_Dense(784, 512)
bn1 = Layer_BatchNormalization(512)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)

dense2 = Layer_Dense(512, 256)
bn2 = Layer_BatchNormalization(256)
activation2 = Activation_ReLU()
dropout2 = Layer_Dropout(0.1)

dense3 = Layer_Dense(256, 128)
bn3 = Layer_BatchNormalization(128)
activation3 = Activation_ReLU()
dropout3 = Layer_Dropout(0.1)

dense4 = Layer_Dense(128, 10)
bn4 = Layer_BatchNormalization(10)
activation4 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-3)

# змінна для збереження попереднього значення втрат на валідації
prev_validation_loss = float('inf')

# Тренування моделі
for epoch in range(20):
    # Пряме поширення
    dense1.forward(train_images)
    bn1.forward(dense1.output)
    activation1.forward(bn1.output)
    dropout1.forward(activation1.output)

    dense2.forward(dropout1.output)
    bn2.forward(dense2.output)
    activation2.forward(bn2.output)
    dropout2.forward(activation2.output)

    dense3.forward(dropout2.output)
    bn3.forward(dense3.output)
    activation3.forward(bn3.output)
    dropout3.forward(activation3.output)

    dense4.forward(dropout3.output)
    activation4.forward(dense4.output)

    # Втрати
    loss = loss_function.calculate(activation4.output, train_labels)

    # Точність
    predictions = np.argmax(activation4.output, axis=1)
    if len(train_labels.shape) == 2:
        y = np.argmax(train_labels, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 1:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {loss:.3f}, ' +
              f'reg_loss: 0), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Зворотне поширення
    loss_function.backward(activation4.output, train_labels)
    activation4.backward(loss_function.dinputs)
    dense4.backward(activation4.dinputs)

    dropout3.backward(dense4.dinputs)
    activation3.backward(dropout3.dinputs)
    bn3.backward(activation3.dinputs)
    dense3.backward(bn3.dinputs)

    dropout2.backward(dense3.dinputs)
    activation2.backward(dropout2.dinputs)
    bn2.backward(activation2.dinputs)
    dense2.backward(bn2.dinputs)

    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    bn1.backward(activation1.dinputs)
    dense1.backward(bn1.dinputs)

    # Оновлення параметрів
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.update_params(dense4)
    optimizer.post_update_params()

    # Рання зупинка
    if epoch % 5 == 0:
        dense1.forward(validation_images)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        activation3.forward(dense3.output)
        dense4.forward(activation3.output)
        activation4.forward(dense4.output)

        loss = loss_function.calculate(activation4.output, validation_labels)
        predictions = np.argmax(activation4.output, axis=1)

        validation_loss = loss_function.calculate(activation4.output, validation_labels)

        # Адаптивна швидкість навчання
        if prev_validation_loss <= validation_loss:
            optimizer.current_learning_rate = optimizer.current_learning_rate * 0.9

        prev_validation_loss = validation_loss

        # Виведення результатів на валідації
        if len(validation_labels.shape) == 2:
            y = np.argmax(validation_labels, axis=1)
        validation_accuracy = np.mean(predictions == y)
        print(f'Validation, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}')
        if validation_accuracy > 0.98:
            print('Early stopping!')
            break


# Збереження моделі (параметрів)
import os
folder_path = "saved_data"
os.makedirs(folder_path, exist_ok=True)

np.save(os.path.join(folder_path, 'dense1_weights.npy'), dense1.weights)
np.save(os.path.join(folder_path, 'dense1_biases.npy'), dense1.biases)
np.save(os.path.join(folder_path, 'dense2_weights.npy'), dense2.weights)
np.save(os.path.join(folder_path, 'dense2_biases.npy'), dense2.biases)
np.save(os.path.join(folder_path, 'dense3_weights.npy'), dense3.weights)
np.save(os.path.join(folder_path, 'dense3_biases.npy'), dense3.biases)
np.save(os.path.join(folder_path, 'dense4_weights.npy'), dense4.weights)
np.save(os.path.join(folder_path, 'dense4_biases.npy'), dense4.biases)
