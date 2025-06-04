# face_recognize.py
import cv2
import numpy as np
from face_detector import FaceDetector
import onnxruntime as ort

class FaceRecognizerONNX:
    def __init__(self, model_path='./models/edgeface_s_gamma_05.onnx', input_size=(112, 112)):
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_img):
        img = cv2.resize(face_img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))[None]  # NCHW
        return img
    
    def get_embedding(self, face_img):
        img_input = self.preprocess(face_img)
        embedding = self.session.run(None, {self.input_name: img_input})[0]
        return embedding[0] / np.linalg.norm(embedding[0])  # L2 normalize