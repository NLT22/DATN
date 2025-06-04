# db_utils.py

import sqlite3
import os

def init_db(db_path="database/face_lock.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Bảng người dùng
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            role TEXT,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Bảng lịch sử mở cửa
    c.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Bảng lưu thư mục ảnh người dùng (chỉ một dòng mỗi user)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            folder_path TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# init_db()

import sqlite3
import random
from datetime import datetime, timedelta

def generate_dummy_logs(db_path="database/face_lock.db", num_logs=50):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    user_ids = [1, 2, 3]  # 👤 Danh sách ID người dùng có sẵn
    results = ['success', 'fail']  # 🎯 Kết quả truy cập

    for _ in range(num_logs):
        user_id = random.choice(user_ids)
        result = random.choice(results)

        # Tạo thời gian ngẫu nhiên trong vòng 30 ngày gần nhất
        days_ago = random.randint(0, 29)
        time_offset = timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        access_time = datetime.now() - time_offset

        c.execute("""
            INSERT INTO access_logs (user_id, access_time, result)
            VALUES (?, ?, ?)
        """, (user_id, access_time.strftime('%Y-%m-%d %H:%M:%S'), result))

    conn.commit()
    conn.close()
    print(f"✅ Đã thêm {num_logs} dòng dữ liệu ngẫu nhiên vào bảng access_logs.")

# Gọi hàm
# generate_dummy_logs()