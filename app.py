## app.py
from flask import Flask, flash, render_template, Response, send_from_directory, request, redirect, jsonify, stream_with_context, session, url_for
import time
from camera import Camera
from functools import wraps
from face_utils import *
import faiss
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
import bcrypt


# from werkzeug.datastructures import MultiDict



app = Flask(__name__)
camera = Camera(detector='haar')
app.secret_key = '12345678' 


UPLOAD_FOLDER = 'uploads'
USER_IMAGE_DIR = 'user_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_IMAGE_DIR, exist_ok=True)

recognizer = FaceRecognizerONNX()

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect("database/face_lock.db")
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ? AND role = 'admin'", (username,))
        result = c.fetchone()
        conn.close()

        if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
            session['admin_logged_in'] = True
            session['admin_username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Sai tên đăng nhập hoặc mật khẩu.')

    return render_template('login.html')

@app.route('/change_password', methods=['GET', 'POST'])
@admin_required
def change_password():
    if request.method == 'POST':
        current = request.form['current_password']
        new = request.form['new_password']
        confirm = request.form['confirm_password']

        if new != confirm:
            return render_template('change_password.html', error='Xác nhận mật khẩu không khớp.')

        conn = sqlite3.connect("database/face_lock.db")
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (session['admin_username'],))
        result = c.fetchone()

        if not result or not bcrypt.checkpw(current.encode('utf-8'), result[0]):
            conn.close()
            return render_template('change_password.html', error='Mật khẩu hiện tại không đúng.')

        new_hash = bcrypt.hashpw(new.encode('utf-8'), bcrypt.gensalt())
        c.execute("UPDATE users SET password = ? WHERE username = ?", (new_hash, session['admin_username']))
        conn.commit()
        conn.close()
        return render_template('change_password.html', success='Đã đổi mật khẩu thành công.')

    return render_template('change_password.html')



@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@admin_required
def index():
    return render_template('index.html')

@app.route('/traffic_monitor')
@admin_required
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
                door = q.get(timeout=0.5)
                yield f"event: door\ndata: {door}\n\n"
            except:
                pass  

            user_id = camera.last_recognized_id or 0
            if user_id != last_user_id:
                last_user_id = user_id
                yield f"event: user\ndata: {user_id}\n\n"
            time.sleep(0.2) 

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

@app.route('/user_images/<path:filename>')
def serve_user_image(filename):
    return send_from_directory('user_images', filename)


@app.route('/manage_users')
@admin_required
def manage_users():
    name_filter = request.args.get('name', '').strip()
    role_filter = request.args.get('role', '')
    start_date = request.args.get('start_date', '')
    page = int(request.args.get('page', 1))
    per_page = 10

    conn = sqlite3.connect('database/face_lock.db')
    c = conn.cursor()

    query = "SELECT id, name, role, username, registered_at FROM users WHERE 1=1"
    params = []

    if name_filter:
        query += " AND name LIKE ?"
        params.append(f"%{name_filter}%")
    
    if role_filter:
        query += " AND role = ?"
        params.append(role_filter)
    
    if start_date:
        query += " AND date(registered_at) >= date(?)"
        params.append(start_date)

    # Đếm số dòng để phân trang
    count_query = f"SELECT COUNT(*) FROM ({query})"
    c.execute(count_query, params)
    total_count = c.fetchone()[0]
    total_pages = (total_count + per_page - 1) // per_page
    has_next = page < total_pages

    # Thêm LIMIT OFFSET
    query += " ORDER BY registered_at DESC LIMIT ? OFFSET ?"
    params += [per_page, (page - 1) * per_page]

    c.execute(query, params)
    user_rows = c.fetchall()

    # Tìm ảnh đầu tiên của mỗi user
    users = []
    for row in user_rows:
        user_id = row[0]
        image_dir = os.path.join('user_images', f'user_{user_id}')
        image_path = None
        if os.path.isdir(image_dir):
            images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if images:
                image_path = f'{image_dir}/{images[0]}'  # VD: user_images/user_3/face_0.jpg

        users.append({
            'id': row[0],
            'name': row[1],
            'role': row[2],
            'username': row[3],
            'registered_at': row[4],
            'image_path': image_path
        })

    conn.close()

    args = request.args.to_dict()
    args.pop('page', None)

    return render_template('manage_users.html',
                           users=users,
                           page=page,
                           per_page=per_page,
                           total_pages=total_pages,
                           has_next=has_next,
                           args=args)


@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    # 1. Xóa thư mục ảnh
    conn = sqlite3.connect('database/face_lock.db')
    c = conn.cursor()
    c.execute("SELECT folder_path FROM user_images WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if row:
        folder_path = row[0]
        if os.path.exists(folder_path):
            import shutil
            shutil.rmtree(folder_path)

    # 2. Xóa khỏi FAISS và id_map
    index = load_index()
    id_map = load_id_mapping()

    # Tìm các index trong id_map khớp với user_id
    ids_to_remove = [i for i, uid in id_map.items() if uid == user_id]
    if ids_to_remove:
        mask = np.ones(index.ntotal, dtype=bool)
        mask[ids_to_remove] = False
        new_vectors = index.reconstruct_n(0, index.ntotal)
        new_vectors = new_vectors[mask]
        
        # Tạo index mới
        dim = new_vectors.shape[1]
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(new_vectors.astype('float32'))
        
        # Cập nhật lại id_map
        new_id_map = {}
        j = 0
        for i in range(index.ntotal):
            if i in ids_to_remove:
                continue
            new_id_map[j] = id_map[i]
            j += 1

        save_index(new_index)
        save_id_mapping(new_id_map)
    else:
        print(f"[WARN] Không tìm thấy embedding cho user_id={user_id} trong FAISS index.")

    # 3. Xóa dữ liệu trong SQLite
    c.execute("DELETE FROM user_images WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM access_logs WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    flash('✅ Đã xóa người dùng thành công!', 'success')
    return redirect(url_for('manage_users'))



@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    conn = sqlite3.connect('database/face_lock.db')
    c = conn.cursor()

    if request.method == 'POST':
        name = request.form['name']
        role = request.form['role']
        username = request.form.get('username') if role == 'admin' else None

        # Kiểm tra nếu là admin thì phải có username
        if role == 'admin' and not username:
            flash('Bạn phải nhập tên đăng nhập cho quản trị viên.', 'danger')
            return redirect(request.url)

        # Kiểm tra username trùng nếu thay đổi username
        c.execute("SELECT username FROM users WHERE id=?", (user_id,))
        current_username = c.fetchone()[0]

        if role == 'admin' and username != current_username:
            c.execute("SELECT id FROM users WHERE username = ?", (username,))
            if c.fetchone():
                conn.close()
                flash('Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.', 'danger')
                return redirect(request.url)

        c.execute("""
            UPDATE users SET name=?, role=?, username=? WHERE id=?
        """, (name, role, username, user_id))
        conn.commit()
        conn.close()
        flash('Cập nhật thông tin người dùng thành công!', 'success')
        return redirect(url_for('manage_users'))

    # GET method
    c.execute("SELECT id, name, role, username FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()

    if row:
        user = dict(id=row[0], name=row[1], role=row[2], username=row[3])
        return render_template('edit_user.html', user=user)
    else:
        flash('Không tìm thấy người dùng!', 'danger')
        return redirect(url_for('manage_users'))


@app.route('/add_user', methods=['GET', 'POST'])
@admin_required
def add_user():
    if request.method == 'POST':
        name = request.form['name']
        role = request.form['role']
        files = request.files.getlist('face_images')

        username = None
        password_hash = None

        if role == 'admin':
            username = request.form.get('username')
            if not username:
                return render_template('add_user.html', error='Bạn phải nhập tên đăng nhập cho quản trị viên.')

            # Kiểm tra username đã tồn tại chưa
            conn = sqlite3.connect("database/face_lock.db")
            c = conn.cursor()
            c.execute("SELECT id FROM users WHERE username = ?", (username,))
            if c.fetchone():
                conn.close()
                return render_template('add_user.html', error='Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.')
            conn.close()

            # Băm mật khẩu mặc định
            import bcrypt
            password_hash = bcrypt.hashpw("0000".encode('utf-8'), bcrypt.gensalt())

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
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding)
                        saved_image_paths.append(img_path)
                    else:
                        print(f"[WARN] Không lấy được embedding cho ảnh: {filename}")
                else:
                    print(f"[WARN] Không detect/align được ảnh: {file.filename}")

        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

            register_user(name, role, mean_embedding, saved_image_paths, username, password_hash)

        else:
            print("[ERROR] Không có embedding nào hợp lệ, không thể đăng ký user.")

        return render_template('add_user.html', success=f"✅ Thêm người dùng '{name}' ({role}) thành công!")

    return render_template('add_user.html')


if __name__ == '__main__':
    app.run(debug=True)
