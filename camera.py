import cv2
from face_detector import FaceDetector
from anti_spoof import AntiSpoof
import numpy as np
import time
import csv
import os

from door_control import set_door_status, auto_close_door, get_door_status
from face_utils import recognize_and_log, get_user_info, increased_crop
from face_recognize import FaceRecognizerONNX

antispoof_threshold = 0.9
cosine_threshold = 0.3
recognition_hold_time = 0.5  
max_face = 1
MIN_LOG_INTERVAL = 30
INPUT_SIZE = 128
scale_up = 1.5

class Camera:
    def __init__(self, detector='haar'):
        self.video = cv2.VideoCapture(0)
        self.face_detector = FaceDetector(detector)
        self.anti_spoof = AntiSpoof('./models/AntiSpoofing_cls2_bbox1.5_sz128_128_best.onnx', input_size=(INPUT_SIZE, INPUT_SIZE))
        self.recognizer = FaceRecognizerONNX()
        self.last_recognized_id = None
        self.last_unlock_time = 0
        self.unlock_delay = 5
        self.real_start_time = None  
        self.latest_frame = None

        # Tạo file log nếu chưa có, KHÔNG có trường fps
        self.log_file = 'log_metrics.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'total_time', 'detect_time', 'antispoof_time', 'recognize_time', 'score', 'label'])

    def log_metrics(self, total_time, detect_time, antispoof_time, recognize_time, score, label):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                total_time,
                detect_time,
                antispoof_time,
                recognize_time,
                score,
                label
            ])

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

                if ret:
                    self.latest_frame = buffer.tobytes()

                time.sleep(0.05)

    def process_frame(self, frame):
        start_total = time.time()
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_detect = time.time()
            bboxes = self.face_detector.detect_faces(rgb_frame)
            detect_time = time.time() - start_detect

            if not bboxes:
                if self.last_recognized_id is not None:
                    print("\U0001f464 No face detected. Resetting identity and starting auto-close timer.")
                    self.last_recognized_id = None
                    self.real_start_time = None
                    auto_close_door()
                total_time = time.time() - start_total
                cv2.putText(frame, f"Total: {total_time*1000:.2f}ms (No face)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                self.log_metrics(
                    total_time,
                    detect_time,
                    None,
                    None,
                    None,
                    'NOFACE'
                )
                return frame

            for bbox in bboxes[:max_face]:
                x1, y1, x2, y2 = bbox

                start_antispoof = time.time()
                face_crop = increased_crop(rgb_frame, bbox, scale_up)
                score = self.anti_spoof.predict(face_crop)
                antispoof_time = time.time() - start_antispoof

                label = 'REAL' if score > antispoof_threshold else 'FAKE'
                color = (0, 255, 0) if label == 'REAL' else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                embedding = None
                recognize_time = None
                elapsed = 0

                if label == 'REAL':
                    now = time.time()
                    if self.real_start_time is None:
                        self.real_start_time = now
                    elapsed = now - self.real_start_time

                    start_recognize = time.time()
                    face_roi = rgb_frame[y1:y2, x1:x2]
                    embedding = self.recognizer.get_embedding(face_roi)

                    if embedding is not None:
                        user_id, result, image_path = recognize_and_log(embedding, cosine_threshold=cosine_threshold, MIN_LOG_INTERVAL=MIN_LOG_INTERVAL)
                        recognize_time = time.time() - start_recognize

                        if user_id and user_id != 0:
                            if elapsed >= recognition_hold_time:
                                if user_id != self.last_recognized_id:
                                    print(f"\U0001f513 Door unlocked for user_id: {user_id} after {elapsed:.2f}s")
                                self.last_recognized_id = user_id
                                set_door_status("OPEN")
                                auto_close_door()
                            user_name, _ = get_user_info(user_id)
                            text = f'{user_name} ({score:.2f})'
                        else:
                            self.last_recognized_id = 0
                            text = 'Unknown'
                            print("\u274c Unknown person — door stays closed")
                        cv2.putText(frame, text, (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    else:
                        print("[WARN] Không lấy được embedding")
                        self.real_start_time = None
                else:
                    self.real_start_time = None

                total_time = time.time() - start_total
                cv2.putText(frame, f"Total: {total_time*1000:.2f}ms", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Detect: {detect_time*1000:.2f}ms", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"AntiSpoof: {antispoof_time*1000:.2f}ms", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if label == 'REAL' and embedding is not None:
                    cv2.putText(frame, f"Recognize: {recognize_time*1000:.2f}ms", 
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Hold: {elapsed:.2f}/{recognition_hold_time}s", 
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                self.log_metrics(
                    total_time,
                    detect_time,
                    antispoof_time,
                    recognize_time,
                    score,
                    label
                )

            return frame

        except Exception as e:
            print("[ERROR] process_frame lỗi:", e)
            total_time = time.time() - start_total
            cv2.putText(frame, f"Total: {total_time*1000:.2f}ms (Error)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            self.log_metrics(
                total_time,
                None,
                None,
                None,
                None,
                'ERROR'
            )
            return frame
