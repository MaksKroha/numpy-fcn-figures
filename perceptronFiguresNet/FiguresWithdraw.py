import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from readers.DatasetReader import DatasetReader


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    mnist_test_file = "../otherFiles/mnist/mnist_test.csv"
    dataset = DatasetReader.getLabelsAndFiguresList(mnist_test_file, np)
    vector_image = dataset[5004, 1:]
    print(dataset[5004, 0])
    image_2d = vector_image.reshape(28, 28)
    # Відображення зображення
    plt.imshow(image_2d, cmap='gray')
    plt.title('Візуалізація зображення з вектора')
    plt.axis('off')
    plt.show()