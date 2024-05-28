import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Button, Label
from tkinter.filedialog import askopenfilename
import os

class Network():
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def learn(self, patterns):
        X = np.array(patterns)
        X_pinv = np.linalg.pinv(X)
        self.weights = X_pinv @ X
        
        np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern, max_iterations=1000000, stability_threshold=1000):
        state = input_pattern.copy()
        stable_iterations = 0

        for iteration in range(max_iterations):
            previous_state = state.copy()

            i = np.random.randint(self.size)
            update = np.dot(self.weights[i], state)
            state[i] = 1 if update >= 0 else -1

            print(f"Iteracja: {iteration} Energia: {self.calculate_energy(state)}")

            if np.array_equal(state, previous_state):
                stable_iterations += 1
            else:
                stable_iterations = 0

            if stable_iterations >= stability_threshold:
                print(f"Sieć jest stabilna")
                break
            
        change_image(state)
        return state
    
    def calculate_energy(self, state):
        return -0.5 * np.sum(self.weights * np.outer(state, state))

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