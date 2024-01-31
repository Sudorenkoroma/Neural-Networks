from keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import pathlib

# Завантаження моделі
model = load_model('my_model.keras')

# Визначення шляху до папки з датасетом
dataset_dir = pathlib.Path('flower_photos')

# Перевірка існування папки датасету
if not dataset_dir.is_dir():
    print(f"Папка {dataset_dir} не знайдена.")
else:
    print(f"Папка знайдена: {dataset_dir}")

# Отримання списку міток класів безпосередньо з назв папок
class_names = [item.name for item in dataset_dir.glob('*/') if item.is_dir()]
print(f"Мітки класів: {class_names}")

# Завантаження зображення, яке потрібно класифікувати
img = Image.open("flower.jpg")
img = img.resize((180, 180))

# Перетворення зображення в масив і додавання виміру пакету
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Виконання передбачення
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Друк результату передбачення
predicted_class = class_names[np.argmax(score)]
probability = 100 * np.max(score)
print(f"На зображенні {predicted_class} ({probability:.2f}% ймовірність)")
