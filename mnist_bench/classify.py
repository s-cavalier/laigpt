import tkinter as tk
import numpy as np
from PIL import Image, ImageOps
from autograd.tensor import Tensor
from mnist_bench.train import build_model


MODEL_PATH = "mnist_model.npz"


class DigitDrawer:
    def __init__(self):
        self.size = 280  # 10× MNIST to draw comfortably
        self.brush_size = 20

        # Create model
        self.model = build_model()
        self.model.load(MODEL_PATH)
        self.model.set_trainability(False)

        # GUI
        self.root = tk.Tk()
        self.root.title("MNIST Digit Classifier")

        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Predict", command=self.predict).pack(side="left")
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side="left")

        self.pred_label = tk.Label(self.root, text="Prediction: None", font=("Arial", 20))
        self.pred_label.pack()

    def draw(self, event):
        x, y = event.x, event.y
        r = self.brush_size // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")

    def clear(self):
        self.canvas.delete("all")
        self.pred_label.config(text="Prediction: None")

    def get_image(self):
        # Get canvas content → convert to grayscale MNIST-style image
        self.canvas.update()
        self.canvas.postscript(file="temp.ps", colormode='color')

        img = Image.open("temp.ps")
        img = img.convert("L")            # grayscale
        img = ImageOps.invert(img)        # MNIST digits are white on black
        img = img.resize((28, 28))        # downsample
        arr = np.array(img).astype(np.float32)
        arr /= 255.0
        arr = arr.reshape(1, 784)         # model expects (1, 784)
        return arr

    def predict(self):
        arr = self.get_image()
        logits = self.model(Tensor(arr))
        pred = int(np.argmax(logits.values))

        self.pred_label.config(text=f"Prediction: {pred}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DigitDrawer().run()
