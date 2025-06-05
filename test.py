import sqlite3
from datetime import datetime

def view_table_contents(db_path="database/face_lock.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    print("\n📋 USERS TABLE:")
    c.execute("SELECT * FROM users")
    users = c.fetchall()
    for row in users:
        print(row)

    # # Kiểm tra user_id = 4
    # c.execute("SELECT id FROM users WHERE id = ?", (4,))
    # if c.fetchone() is None:
    #     print("Lỗi: user_id = 4 không tồn tại trong bảng users! Vui lòng thêm người dùng trước.")
    # else:
    #     # Thêm bản ghi vào access_logs
    #     try:
    #         # c.execute("INSERT INTO access_logs (user_id, access_time, result) VALUES (?, ?, ?)", 
    #         #           (4, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'success'))
    #         # conn.commit()
    #         c.execute("INSERT INTO access_logs (user_id, result) VALUES (?, ?)", (4,'success'))
    #         # conn.commit()
    #         print("✅ Đã thêm bản ghi vào access_logs.")
    #     except sqlite3.IntegrityError as e:
    #         print(f"Lỗi khi thêm vào access_logs: {e}")

    print("\n📋 ACCESS_LOGS TABLE (50 bản ghi mới nhất):")
    c.execute("SELECT * FROM access_logs ORDER BY access_time DESC LIMIT 50")
    logs = c.fetchall()
    for row in logs:
        print(row)

    print("\n📋 USER_IMAGES TABLE:")
    c.execute("SELECT * FROM user_images")
    images = c.fetchall()
    for row in images:
        print(row)

    conn.close()


import pickle

def view_id_to_user():

    # Đường dẫn đến file ánh xạ index → user_id
    id_map_path = 'embeddings/id_to_user.pkl'

    # Load ánh xạ
    with open(id_map_path, 'rb') as f:
        id_map = pickle.load(f)

    print("🔍 ID Map (index → user_id):")
    for index, user_id in id_map.items():
        print(f"  Index {index} → User ID {user_id}")

import faiss
import numpy as np

def view_index_bin():
    # Đường dẫn tới FAISS index
    index_path = 'faiss_index/index.bin'

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Thông tin cơ bản
    print(f"📌 Số vector: {index.ntotal}")
    dim = index.d  # số chiều vector
    print(f"📐 Chiều vector: {dim}")

    # Nếu muốn in toàn bộ vectors:
    vectors = index.reconstruct_n(0, index.ntotal)  # numpy array
    print("📊 Các vector embeddings:")
    for i, v in enumerate(vectors):
        print(f"Vector {i} (length={np.linalg.norm(v):.4f}): {v[:5]}...")  # chỉ in 5 giá trị đầu cho gọn


if __name__ == "__main__":
    view_table_contents()
    view_id_to_user()
    view_index_bin()