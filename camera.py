# camera.py

import cv2
from face_detector import FaceDetector
from anti_spoof import AntiSpoof
import numpy as np
import time

from door_control import set_door_status, auto_close_door, get_door_status    # ⬅️ import từ door_control

threshold = 0.75

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w / 2, y + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(real_w, int(x + l * bbox_inc))
    y2 = min(real_h, int(y + l * bbox_inc))
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img,
                             y1 - y, int(l * bbox_inc - y2 + y),
                             x1 - x, int(l * bbox_inc) - x2 + x,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

class Camera:
    def __init__(self, detector='haar'):
        self.video = cv2.VideoCapture(1)
        self.face_detector = FaceDetector(detector)
        self.anti_spoof = AntiSpoof('./models/AntiSpoofing_cls2_bbox1.5_sz128_128_best.onnx')

        self.last_unlock_time = 0
        self.unlock_delay = 5

    def gen_frames(self):
        while True:
            success, frame = self.video.read()
            if not success:
                print("[ERROR] Không đọc được frame từ camera")
                break
            else:
                frame = cv2.flip(frame, 1)
                annotated_frame = self.process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    
    def process_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = self.face_detector.detect_faces(rgb_frame)

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                face_img = increased_crop(rgb_frame, bbox, 1.5)
                score = self.anti_spoof.predict(face_img)
                label = 'REAL' if score > threshold else 'FAKE'
                color = (0, 255, 0) if label == 'REAL' else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if label == 'REAL':
                    if get_door_status() == "CLOSED":
                        print("🔓 Door Unlocked")
                        set_door_status("OPEN")

                    auto_close_door()  # reset lại timeout mỗi lần thấy REAL

            return frame
        except Exception as e:
            print("[ERROR] process_frame lỗi:", e)
            return frame