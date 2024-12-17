import os
import sys
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageDraw
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing import image

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Salfronsii's Program")
        self.root.resizable(False, False)
        self.canvas_size = 280  # Размер холста 280x280
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack()

        # Создаем изображение и объект для рисования
        self.image = Image.new("RGB", (28, 28), "black")  # Черный фон
        self.draw = ImageDraw.Draw(self.image)

        # Переменные для рисования
        self.last_x, self.last_y = None, None

        # Привязываем события мыши к методам
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Кнопка для сохранения изображения
        self.save_button = Button(root, text="Сохранить", command=self.save_image)
        self.save_button.pack()

        # Кнопка для очистки холста
        self.clear_button = Button(root, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack()

        # Кнопка для предсказания
        self.predict_button = Button(root, text="Предсказать", command=self.predict_digit)
        self.predict_button.pack()

        # Метка для отображения предсказанной цифры
        self.prediction_label = Label(root, text="", font=("Helvetica", 24))
        self.prediction_label.pack()

        # Загрузка модели
        self.model = self.load_modell('my_model.h5')

    def load_modell(self, model_name):
        """ Загружает модель, учитывая, что она может быть упакована в исполняемый файл. """
        model_path = self.resource_path(model_name)
        return load_model(model_path)

    def resource_path(self, relative_path):
        """ Получает абсолютный путь к ресурсу, который будет работать как в режиме разработки, так и в собранном приложении. """
        try:
            # PyInstaller создает временную папку и сохраняет путь к ней в _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_line(self, event):
        if self.last_x is not None and self.last_y is not None:
            # Рисуем на холсте
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill='white', width=2)  # Белый цвет
            # Рисуем на изображении с учетом масштабирования
            scale = 10  # Масштаб для преобразования координат
            self.draw.line(
                [self.last_x // scale, self.last_y // scale, event.x // scale, event.y // scale],
                fill='white', width=1  # Белый цвет
            )
            self.last_x, self.last_y = event.x, event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def save_image(self):
        self.image.save("image.png")

    def clear_canvas(self):
        # Очищаем холст
        self.canvas.delete("all")
        # Очищаем изображение
        self.image = Image.new("RGB", (28, 28), "black")  # Черный фон
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="")  # Очищаем текст метки

    def predict_digit(self):
        # Загрузка изображения
        img_path = 'image.png'
        img = image.load_img(img_path, target_size=(28, 28))  # Загружаем изображение
        img = img.convert('L')  # Преобразуем в градации серого
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Добавляем размерность для батча
        img = img / 255.0  # Нормализация
        # Предсказание
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)

        # Обновляем текст метки с предсказанной цифрой
        self.prediction_label.config(text=f'Предсказанная цифра: {predicted_class}')


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

