# camera.py

import cv2
from face_detector import FaceDetector
from anti_spoof import AntiSpoof
import numpy as np
import time

from door_control import set_door_status, auto_close_door, get_door_status    # ⬅️ import từ door_control
from face_utils import recognize_and_log, get_user_info, increased_crop
from face_recognize import FaceRecognizerONNX

antispoof_threshold = 0.75

class Camera:
    def __init__(self, detector='haar'):
        self.video = cv2.VideoCapture(1)
        self.face_detector = FaceDetector(detector)
        self.anti_spoof = AntiSpoof('./models/AntiSpoofing_cls2_bbox1.5_sz128_128_best.onnx')

        self.recognizer = FaceRecognizerONNX()
        self.last_recognized_id = None  

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

            if not bboxes:
                # Không phát hiện khuôn mặt → reset trạng thái nhận diện và bắt đầu đếm ngược tự động khóa
                if self.last_recognized_id is not None:
                    print("👤 No face detected. Resetting identity and starting auto-close timer.")
                    self.last_recognized_id = None
                    auto_close_door()
                return frame  # return sớm nếu không có bbox

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox

                # Face crop mở rộng chỉ dùng cho anti-spoof
                face_crop = increased_crop(rgb_frame, bbox, 1.5)
                score = self.anti_spoof.predict(face_crop)
                label = 'REAL' if score > antispoof_threshold else 'FAKE'
                color = (0, 255, 0) if label == 'REAL' else (0, 0, 255)

                # Vẽ khung và label anti-spoof
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if label == 'REAL':
                    face_roi = rgb_frame[y1:y2, x1:x2]
                    embedding = self.recognizer.get_embedding(face_roi)

                    if embedding is not None:
                        user_id, result, image_path = recognize_and_log(embedding)

                        if user_id and user_id != 0: 
                            self.last_recognized_id = user_id

                            print(f"🔓 Door (re)unlocked for user_id: {user_id}")
                            set_door_status("OPEN")
                            auto_close_door()

                            user_name, _ = get_user_info(user_id)
                            text = f'{user_name} ({score:.2f})'
                        else:
                            self.last_recognized_id = 0
                            text = 'Unknown'
                            print("🚫 Unknown person — door stays closed")


                        cv2.putText(frame, text, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    else:
                        print("[WARN] Không lấy được embedding")

            return frame

        except Exception as e:
            print("[ERROR] process_frame lỗi:", e)
            return frame