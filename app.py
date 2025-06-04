## app.py
from flask import Flask, render_template, Response, send_from_directory, request, redirect, jsonify, stream_with_context
import time
from camera import Camera
import db_utils, face_utils

from face_utils import register_user
import numpy as np
import sqlite3
from collections import defaultdict
from datetime import datetime
import calendar
import math
import os
import cv2
from face_recognize import FaceRecognizerONNX
from face_detector import FaceDetector
from werkzeug.utils import secure_filename
from door_control import get_door_status
from door_control import get_door_event_queue

# from werkzeug.datastructures import MultiDict


app = Flask(__name__)
camera = Camera(detector='haar')

UPLOAD_FOLDER = 'uploads'
USER_IMAGE_DIR = 'user_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_IMAGE_DIR, exist_ok=True)

recognizer = FaceRecognizerONNX()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/traffic_monitor')
def traffic_monitor():
    # Lấy các tham số lọc và phân trang
    name_filter = request.args.get('name', '').strip()
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    role_filter = request.args.get('role', '').strip()
    group_by = request.args.get('group_by', 'day')
    page = int(request.args.get('page', 1))
    per_page = 50

    # Câu truy vấn chính
    query = """
        SELECT u.name, u.role, a.access_time, a.result
        FROM access_logs a
        JOIN users u ON a.user_id = u.id
        WHERE 1=1
    """
    params = []

    if name_filter:
        query += " AND u.name LIKE ?"
        params.append(f"%{name_filter}%")

    if start_date:
        query += " AND DATE(a.access_time) >= ?"
        params.append(start_date)

    if end_date:
        query += " AND DATE(a.access_time) <= ?"
        params.append(end_date)

    if role_filter:
        query += " AND u.role = ?"
        params.append(role_filter)

    query += " ORDER BY a.access_time DESC"

    # Kết nối DB và thực hiện truy vấn
    conn = sqlite3.connect("database/face_lock.db")
    c = conn.cursor()
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    # Tính thống kê biểu đồ và phân trang
    logs = []
    date_counter = defaultdict(int)

    for row in rows:
        name, role, access_time, result = row
        logs.append({
            "name": name,
            "role": role,
            "access_time": access_time,
            "result": result
        })

        dt = datetime.strptime(access_time, "%Y-%m-%d %H:%M:%S")

        if group_by == "week":
            label = f"Tuần {dt.isocalendar()[1]:02d}"
        elif group_by == "month":
            label = f"Tháng {dt.strftime('%m')}"
        else:
            label = dt.strftime("%d/%m")

        date_counter[label] += 1

    chart_labels = sorted(date_counter.keys())
    chart_counts = [date_counter[label] for label in chart_labels]

    # Phân trang thủ công
    total_records = len(logs)
    total_pages = math.ceil(total_records / per_page) if per_page else 1
    start = (page - 1) * per_page
    end = start + per_page
    paginated_logs = logs[start:end]
    has_next = page < total_pages

    args = request.args.to_dict(flat=True)
    args.pop('page', None)  # Xoá page

    return render_template(
        "traffic.html",
        logs=paginated_logs,
        chart_labels=chart_labels,
        chart_counts=chart_counts,
        page=page,
        per_page=per_page,
        has_next=has_next,
        total_pages=total_pages,
        args=args  # truyền dict đã loại bỏ page
    )

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

@app.route('/door_status')
def door_status():
    return get_door_status()

@app.route('/video_feed')
def video_feed():
    return Response(camera.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sse/updates')
def sse_updates():
    def event_stream():
        last_user_id = None
        q = get_door_event_queue()

        while True:
            try:
                # Check door status update
                door = q.get(timeout=0.5)
                yield f"event: door\ndata: {door}\n\n"
            except:
                pass  # No new door event

            # Check if user ID changed
            user_id = camera.last_recognized_id or 0
            if user_id != last_user_id:
                last_user_id = user_id
                yield f"event: user\ndata: {user_id}\n\n"
            time.sleep(0.2)  # prevent CPU overuse

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

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
                        # Chuẩn hóa embedding trước khi thêm vào danh sách
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding)
                        saved_image_paths.append(img_path)
                    else:
                        print(f"[WARN] Không lấy được embedding cho ảnh: {filename}")
                else:
                    print(f"[WARN] Không detect/align được ảnh: {file.filename}")

        if embeddings:
            # Tính trung bình các embedding và chuẩn hóa lại
            mean_embedding = np.mean(embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            register_user(name, role, mean_embedding, saved_image_paths)
        else:
            print("[ERROR] Không có embedding nào hợp lệ, không thể đăng ký user.")

        return redirect('/add_user')

    return render_template('add_user.html')

if __name__ == '__main__':
    app.run(debug=True)
