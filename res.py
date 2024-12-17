import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing import image

# Загрузка модели
model = load_model('my_model.h5')


# Загрузка изображения
img_path = 'image.png'
img = image.load_img(img_path, target_size=(28, 28))  # Загружаем изображение
img = img.convert('L')  # Преобразуем в градации серого
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)  # Добавляем размерность для батча
img = img / 255.0  # Нормализация

# Предсказание
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

print(f'Предсказанная цифра: {predicted_class}')