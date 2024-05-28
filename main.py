import numpy as np
from PIL import Image, ImageTk
from random import random
from tkinter import Tk, Canvas, Button, PhotoImage, Label
from tkinter.filedialog import askopenfilename
import os

class Network():
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def learn(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)

        np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern, max_iterations=1000000):
        state = input_pattern.copy()

        for _ in range(max_iterations):
            i = np.random.randint(self.size)
            update = np.dot(self.weights[i], state)
            state[i] = 1 if update >= 0 else -1
            
        change_image(state)
        return state
            

def binarize_image(file_name):
    image = Image.open(file_name)
    array = np.array(image.convert("1"))
    return (array * 2 - 1).flatten()

def unbinarize_image(bitmap):
    unbinarized_bitmap = []
    for i in bitmap:
        unbinarized_bitmap.append(1 if i == 1 else 0)

    array = np.asarray(unbinarized_bitmap).reshape((28, 28)) * 255
    image = Image.fromarray(array.astype(np.uint8))

    return image

def select_image():
    global test_image
    filename = askopenfilename()
    test_image = binarize_image(filename)
    change_image(test_image)
    
def change_image(bitmap):
    label_image = ImageTk.BitmapImage(unbinarize_image(bitmap).resize((280, 280), resample=Image.Resampling.NEAREST).convert("1"))
    image.img = label_image
    image.config(image=label_image)

def start_prediction():
    network.predict(test_image)

file_names = []
for filename in os.listdir("./wzorce"):
    if filename.endswith(".png"):
        file_names.append(filename)
        
binarized = [binarize_image(f"./wzorce/{name}") for name in file_names]
test_image = []

network = Network(binarized[0].size)
root = Tk()
image = Label(root)

def main():
    network.learn(binarized)

    select_button = Button(root, text="Wybierz obraz", command=select_image)
    select_button.pack()

    image.pack()

    start_button = Button(root, text="Rozpocznij dopasowywanie", command=start_prediction)
    start_button.pack()

    root.mainloop()


if __name__ == "__main__":
    main()