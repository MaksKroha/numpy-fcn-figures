import tkinter as tk
import numpy as np
from optimizations.BatchNorm import BatchNormPerceptron as BatchNormP
from readers.ParamsReader import ParamsReader
from ForwardPass import ForwardPass
from activations.CrossEntropy import CrossEntropy


class DrawGrid:
    def __init__(self, master):
        self.perc_params_file = "../otherFiles/perceptron_params.json"
        self.batch_params_file = "../otherFiles/batchnorm_params.json"
        self.batch_properties_file = "../otherFiles/batchnorm_properties.json"

        self.master = master
        self.master.title("Малювання на сітці 28x28")

        # Параметри сітки
        self.rows = 28
        self.columns = 28
        self.cell_size = 20  # Розмір кожної клітинки

        # Масив для збереження малюнка
        self.grid = np.zeros((self.rows, self.columns), dtype=np.float32)

        # Canvas для відображення сітки
        self.canvas = tk.Canvas(master, width=self.columns * self.cell_size, height=self.rows * self.cell_size, bg='white')
        self.canvas.grid(row=0, column=0, rowspan=4)

        # Намалювати сітку
        self.draw_grid()

        # Додавання події малювання
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # Додавання кнопок
        self.btn_save = tk.Button(master, text="Зберегти", command=self.save_drawing)
        self.btn_save.grid(row=0, column=1, sticky="w")

        self.btn_clear = tk.Button(master, text="Очистити", command=self.clear_canvas)
        self.btn_clear.grid(row=1, column=1, sticky="w")

        self.btn_convert = tk.Button(master, text="Перетворити у вектор", command=self.convert_to_vector)
        self.btn_convert.grid(row=2, column=1, sticky="w")

        # Поле для відображення результату
        self.result_label = tk.Label(master, text="Очікуваний результат:")
        self.result_label.grid(row=3, column=1, sticky="w")

        self.result_var = tk.StringVar()
        self.result_entry = tk.Entry(master, textvariable=self.result_var, width=10)
        self.result_entry.grid(row=3, column=1, sticky="e")

    def draw_grid(self):
        """Малювання сітки на Canvas."""
        for i in range(self.rows):
            for j in range(self.columns):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='gray')

    def perceptronTest(self, vector):
        BatchNormP.installFromFile(self.batch_params_file, self.batch_properties_file, np)
        params = ParamsReader.readParams(self.perc_params_file)
        summa = ForwardPass.forwardTestPerceptron(vector, params)
        cross = CrossEntropy(summa, [0], np)
        probabilities = cross.getProbabilities()[0].tolist()
        print(probabilities)
        max_val, number = 0.0, 0
        for i in range(len(probabilities)):
            if probabilities[i] > max_val:
                max_val = probabilities[i]
                number = i
        return f"{number}  {int(max_val * 100)}%"

    def paint(self, event):
        """Малювання пензлем на сітці."""
        x, y = event.x, event.y

        # Визначення центрального рядка та стовпчика
        center_row = y // self.cell_size
        center_col = x // self.cell_size

        # Перевірка, що координати не виходять за межі сітки
        if 0 <= center_row < self.rows and 0 <= center_col < self.columns:
            # Малювання товстим пензлем
            self.draw_brush(center_row, center_col, "black")

    def draw_brush(self, row, col, color):
        """Малювання квадратом із заданим кольором."""
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

        # Оновлення масиву
        self.grid[row, col] = 1.0

    def generate_augmented_grid(self):
        """Генерація нової матриці з пікселями зі значенням 0.5 навколо пікселів зі значенням 1."""
        augmented_grid = np.copy(self.grid)

        for i in range(self.rows):
            for j in range(self.columns):
                if self.grid[i, j] == 1.0:
                    # Додавання пікселів зі значенням 0.5 тільки з боків
                    if i > 0 and augmented_grid[i - 1, j] == 0:  # Верх
                        augmented_grid[i - 1, j] = 0.5
                    if i < self.rows - 1 and augmented_grid[i + 1, j] == 0:  # Низ
                        augmented_grid[i + 1, j] = 0.5
                    if j > 0 and augmented_grid[i, j - 1] == 0:  # Ліворуч
                        augmented_grid[i, j - 1] = 0.5
                    if j < self.columns - 1 and augmented_grid[i, j + 1] == 0:  # Праворуч
                        augmented_grid[i, j + 1] = 0.5

        return augmented_grid

    def save_drawing(self):
        """Збереження малюнка у файл."""
        augmented_grid = self.generate_augmented_grid()
        np.savetxt("drawing.txt", augmented_grid, fmt='%0.1f')
        print("Малюнок збережено як drawing.txt")

    def convert_to_vector(self):
        """Перетворення матриці в 784-вимірний вектор та вивід його в консоль."""
        # Генерація нової матриці з пікселями зі значенням 0.5
        augmented_grid = self.generate_augmented_grid()

        # Перетворення матриці 28x28 в одновимірний вектор
        vector = augmented_grid.flatten()
        print("Вектор (784 пікселів):")

        # Оновлення поля результату
        self.result_var.set(self.perceptronTest(vector))

        # Оновлення сітки та відображення нового малюнка
        self.grid = augmented_grid
        self.redraw_grid()

    def redraw_grid(self):
        """Перемальовування сітки на Canvas згідно з новою матрицею."""
        self.canvas.delete("all")
        self.draw_grid()  # Відновлюємо сітку
        for i in range(self.rows):
            for j in range(self.columns):
                if self.grid[i, j] == 1.0:
                    self.draw_brush(i, j, "#1f1f1f")  # Колір для пікселів зі значенням 1
                elif self.grid[i, j] == 0.5:
                    self.draw_brush(i, j, "#3d3d3d")  # Колір для пікселів зі значенням 0.5

    def clear_canvas(self):
        """Очищення сітки та Canvas."""
        self.canvas.delete("all")
        self.grid = np.zeros((self.rows, self.columns), dtype=np.float32)
        self.draw_grid()
        self.result_var.set("")


# Запуск програми
root = tk.Tk()
app = DrawGrid(root)
root.mainloop()
