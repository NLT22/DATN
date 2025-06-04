## app.py
from flask import Flask, render_template, Response, send_from_directory, request, redirect
from camera import Camera
import db_utils, face_utils
import numpy as np
import sqlite3
import os
import cv2
from face_recognize import FaceRecognizerONNX
from face_detector import FaceDetector
from werkzeug.utils import secure_filename
from door_control import get_door_status

app = Flask(__name__)
camera = Camera(detector='yunet')  # yunet/haar

UPLOAD_FOLDER = 'uploads'
USER_IMAGE_DIR = 'user_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_IMAGE_DIR, exist_ok=True)

recognizer = FaceRecognizerONNX()

@app.route('/')
def index():
    user_id = camera.last_recognized_id or 0
    image_url = f'/recognized_identity_image/{user_id}'
    return render_template('index.html', door_status=get_door_status(), recognized_image=image_url)

@app.route('/current_user_id')
def current_user_id():
    user_id = camera.last_recognized_id or 0
    return {'user_id': user_id}

@app.route('/traffic_monitor')
def traffic_monitor():
    return render_template('traffic.html')

@app.route('/recognized_identity_image/<int:user_id>')
def recognized_identity_image(user_id):
    if user_id == 0:
        return send_from_directory(USER_IMAGE_DIR, 'Unknown.png')

    user_folder = os.path.join(USER_IMAGE_DIR, f'user_{user_id}')
    if os.path.exists(user_folder):
        images = sorted([f for f in os.listdir(user_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if images:
            return send_from_directory(user_folder, images[0])
    return send_from_directory(USER_IMAGE_DIR, 'Unknown.png')


@app.route('/identity_image')
def identity_image():
    return send_from_directory('Identity', 'identity_image.jpeg')

@app.route('/door_status')
def door_status():
    return get_door_status()

@app.route('/video_feed')
def video_feed():
    return Response(camera.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        name = request.form['name']
        role = request.form['role']
        files = request.files.getlist('face_images')

        embeddings = []
        saved_image_paths = []
        from face_retina_detector import RetinaFaceDetector
        detector = RetinaFaceDetector(device='cpu')

        for file in files:
            if file and file.filename:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                aligned = detector.detect_and_align(img)

                if aligned is not None:
                    filename = secure_filename(file.filename)
                    img_path = os.path.join(UPLOAD_FOLDER, filename)
                    cv2.imwrite(img_path, aligned)

                    embedding = recognizer.get_embedding(aligned)
                    if embedding is not None:
                        embeddings.append(embedding)
                        saved_image_paths.append(img_path)
                    else:
                        print(f"[WARN] Không lấy được embedding cho ảnh: {filename}")
                else:
                    print(f"[WARN] Không detect/align được ảnh: {file.filename}")

        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            register_user(name, role, mean_embedding, saved_image_paths)
        else:
            print("[ERROR] Không có embedding nào hợp lệ, không thể đăng ký user.")

        return redirect('/add_user')

    return render_template('add_user.html')


def register_user(name, role, embedding, image_paths):
    # 1. Lưu user vào DB
    conn = sqlite3.connect("database/face_lock.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (name, role) VALUES (?, ?)", (name, role))
    user_id = c.lastrowid
    conn.commit()

    # 2. Tạo thư mục riêng cho user
    user_folder = os.path.join(USER_IMAGE_DIR, f'user_{user_id}')
    os.makedirs(user_folder, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        filename = f'face_{idx}.jpg'
        new_path = os.path.join(user_folder, filename)
        os.rename(img_path, new_path)

    # Chỉ lưu thư mục chứa ảnh vào bảng user_images
    c.execute("INSERT INTO user_images (user_id, folder_path) VALUES (?, ?)", (user_id, user_folder))

    conn.commit()
    conn.close()

    # 4. Thêm embedding vào FAISS
    index = face_utils.load_index()
    id_map = face_utils.load_id_mapping()

    embedding = np.array([embedding]).astype('float32')
    index.add(embedding)

    idx = index.ntotal - 1
    id_map[idx] = user_id

    face_utils.save_index(index)
    face_utils.save_id_mapping(id_map)

    print(f"✅ Registered {name} with user_id={user_id}, images={len(image_paths)}")


if __name__ == '__main__':
    app.run(debug=True)
