import faiss
import numpy as np
import pickle
import os
import sqlite3
import cv2
from datetime import datetime, timedelta

# Lưu thời điểm nhận diện cuối cùng theo user_id
last_logged_times = {}

# Khoảng thời gian tối thiểu giữa 2 lần log liên tiếp cho cùng user (giây)
MIN_LOG_INTERVAL = 10  

FAISS_INDEX_PATH = "faiss_index/index.bin"
ID_MAPPING_PATH = "embeddings/id_to_user.pkl"
USER_IMAGE_DIR = 'user_images'

# Ngưỡng Cosine similarity (thay vì Euclidean distance)
cosine_threshold = 0.5  # Giá trị từ 0 đến 1, càng gần 1 càng tương đồng

def create_faiss_index(dim=512):
    # Sử dụng IndexFlatIP cho Cosine similarity (Inner Product)
    index = faiss.IndexFlatIP(dim) 
    return index

def save_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

def load_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        return create_faiss_index()

def save_id_mapping(id_map):
    with open(ID_MAPPING_PATH, "wb") as f:
        pickle.dump(id_map, f)

def load_id_mapping():
    if os.path.exists(ID_MAPPING_PATH):
        with open(ID_MAPPING_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return {}
    
def get_user_info(user_id):
    conn = sqlite3.connect("database/face_lock.db")
    c = conn.cursor()
    c.execute("SELECT name, role FROM users WHERE id=?", (user_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0], result[1]
    else:
        return "Unknown", ""

def recognize_and_log(embedding):
    index = load_index()
    id_map = load_id_mapping()
    
    # Chuẩn hóa embedding đầu vào
    embedding = np.array([embedding]).astype('float32')
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    # Tìm kiếm vector gần nhất (Inner Product cho Cosine similarity)
    D, I = index.search(embedding, 1)
    if D[0][0] > cosine_threshold:  # So sánh với ngưỡng Cosine
        user_id = id_map[I[0][0]]
        result = "success"

        # Lấy ảnh đầu tiên trong thư mục user_images/user_{user_id}
        user_folder = os.path.join(USER_IMAGE_DIR, f'user_{user_id}')
        first_image = None
        if os.path.exists(user_folder):
            images = sorted([f for f in os.listdir(user_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if images:
                first_image = os.path.join(user_folder, images[0])

        # Kiểm tra khoảng thời gian từ lần log trước
        now = datetime.now()
        last_time = last_logged_times.get(user_id)
        if not last_time or (now - last_time).total_seconds() >= MIN_LOG_INTERVAL:
            conn = sqlite3.connect("database/face_lock.db")
            c = conn.cursor()
            c.execute("INSERT INTO access_logs (user_id, access_time, result) VALUES (?, ?, ?)", 
                            (user_id, now.strftime('%Y-%m-%d %H:%M:%S'), result))
            conn.commit()
            conn.close()
            last_logged_times[user_id] = now  # Cập nhật thời gian log cuối cùng
        else:
            print(f"[INFO] Đã nhận diện user_id {user_id} gần đây, không ghi log.")

    else:
        user_id = None
        result = "failed"
        first_image = './user_images/Unknown.png'

    return user_id, result, first_image

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
    index = load_index()
    id_map = load_id_mapping()

    # Chuẩn hóa embedding trước khi thêm
    embedding = np.array([embedding]).astype('float32')
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    index.add(embedding)

    idx = index.ntotal - 1
    id_map[idx] = user_id

    save_index(index)
    save_id_mapping(id_map)

    print(f"✅ Registered {name} with user_id={user_id}, images={len(image_paths)}")

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