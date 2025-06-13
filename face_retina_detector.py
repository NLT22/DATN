# face_retina_detector.py

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class RetinaFaceDetector:
    def __init__(self, device='cpu'):
        from insightface.app import FaceAnalysis 
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def detect_and_align(self, image, output_size=(112, 112)):
        faces = self.app.get(image)

        if not faces:
            print("[INFO] Không phát hiện khuôn mặt.")
            return None

        face = faces[0]  # chỉ lấy khuôn mặt đầu tiên
        landmarks = face.kps.astype(np.float32)

        # Chuẩn 5 điểm
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
            print("[WARN] Không thể tính affine matrix.")
            return None

        aligned = cv2.warpAffine(image, M, output_size, flags=cv2.INTER_LINEAR)
        return aligned
