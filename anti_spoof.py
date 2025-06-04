import cv2
import onnxruntime as ort
import numpy as np

class AntiSpoof:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.resize(img, (128, 128))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img):
        input_tensor = self.preprocess(img)
        output = self.session.run(None, {self.input_name: input_tensor})
        prob = self.softmax(output[0])[0][0]
        return prob

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)
