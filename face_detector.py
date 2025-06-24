import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, detector_type='haar'):
        self.detector_type = detector_type
        if detector_type == 'haar':
            self.detector = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
        elif detector_type == 'yunet':
            self.detector = cv2.FaceDetectorYN.create(
                model='models/face_detection_yunet_2023mar.onnx',
                config='',
                input_size=(300, 300))
        else:
            raise ValueError("Unsupported detector type")

    def detect_faces(self, image):
        if self.detector_type == 'haar':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            return [(x, y, x + w, y + h) for (x, y, w, h) in faces]

        elif self.detector_type == 'yunet':
            self.detector.setInputSize((image.shape[1], image.shape[0]))
            _, faces = self.detector.detect(image)
            bboxes = []
            if faces is not None:
                for face in faces:
                    if len(face) >= 4:
                        x, y, w, h = face[:4].astype(int)
                        bboxes.append((x, y, x + w, y + h))
            return bboxes

    def detect_and_align(self, image, output_size=(112, 112)):
        if self.detector_type != 'yunet':
            raise NotImplementedError("detect_and_align requires YUNet for landmarks")

        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            print("[INFO] No faces detected.")
            return None

        face = faces[0]
        if face.shape[0] < 14:  # 4 bbox + 10 landmark (5 điểm x,y)
            print(f"[WARN] Landmark không đầy đủ. Shape: {face.shape}")
            return None

        try:
            landmarks = face[4:14].reshape((5, 2)).astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Không thể reshape landmarks: {e}")
            return None

        ref_landmarks = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

        ref_landmarks *= (output_size[0] / 112.0)

        M, _ = cv2.estimateAffinePartial2D(landmarks, ref_landmarks, method=cv2.LMEDS)
        if M is None:
            print("[WARN] estimateAffinePartial2D failed.")
            return None

        aligned = cv2.warpAffine(image, M, output_size, flags=cv2.INTER_LINEAR)
        return aligned
