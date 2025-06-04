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