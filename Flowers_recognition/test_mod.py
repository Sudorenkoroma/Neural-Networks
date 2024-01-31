from keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pathlib

model = load_model('my_model.keras')

dataset_dir = os.path.abspath('flower_photos')
dataset_dir = pathlib.Path(dataset_dir)



if not dataset_dir.is_dir():
    print(f"Папка {dataset_dir} не знайдена.")
else:
    print(f"Папка знайдена: {dataset_dir}")


image_count = len(list(dataset_dir.glob("*/*.jpg")))
print(f"Всего изображений: {image_count}")

batch_size = 32
img_width = 180
img_height = 180


train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)


class_names = train_ds.class_names

img = Image.open("sunflower.jpg")
img = img.resize((180, 180))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# make predictions
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# print inference result
print("На изображении скорее всего {} ({:.2f}% вероятность)".format(class_names[np.argmax(score)],
    100 * np.max(score)))

# show the image itself
img.show()
