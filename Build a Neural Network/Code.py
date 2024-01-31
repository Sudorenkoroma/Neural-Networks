from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from PIL import Image

# Завантаження та підготовка даних MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Створення моделі
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Тренування моделі
model.fit(train_images, train_labels, epochs=10, batch_size=128)

# Оцінка моделі
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nAccuracy on test set:', test_acc)

try:
    img = Image.open("test_img.png").convert('L')
except IOError:
    print("Не вдалося відкрити або знайти зображення 'test_img.png'")
    img = None

# Якщо зображення вдалося відкрити, продовжуйте обробку
if img is not None:
    img = np.invert(img.resize((28, 28)))
    img = np.array(img).ravel()

    prediction = model.predict(np.array([img]))
    predicted_digit = np.argmax(prediction)
    print("Prediction for test image:", predicted_digit)

else:
    print("Помилка: зображення не було завантажене.")

model.save('my_model.keras')