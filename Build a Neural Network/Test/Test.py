from keras.models import load_model
import numpy as np
from PIL import Image
model = load_model('my_model.keras')

test_data = Image.open("test.png").convert('L')
test_data = np.invert(test_data.resize((28, 28)))
test_data = np.array(test_data).ravel()
# test_data = np.array(test_data)
if test_data.ndim == 1:
    test_data = np.expand_dims(test_data, axis=0)

# Використовуємо модель для передбачення
predictions = model.predict(test_data)
predicted_digit = np.argmax(predictions)
print("Prediction for test image:", predicted_digit)
