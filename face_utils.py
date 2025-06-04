# face_utils.py
import faiss
import numpy as np
import pickle
import os
import sqlite3
import cv2

FAISS_INDEX_PATH = "faiss_index/index.bin"
ID_MAPPING_PATH = "embeddings/id_to_user.pkl"
USER_IMAGE_DIR = 'user_images'

eculid_distance = 1

def create_faiss_index(dim=512):
    index = faiss.IndexFlatL2(dim) 
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
    embedding = np.array([embedding]).astype('float32')

    D, I = index.search(embedding, 1)
    if D[0][0] < eculid_distance :
        user_id = id_map[I[0][0]]
        result = "success"

        # Lấy ảnh đầu tiên trong thư mục user_images/user_{user_id}
        user_folder = os.path.join(USER_IMAGE_DIR, f'user_{user_id}')
        first_image = None
        if os.path.exists(user_folder):
            images = sorted([f for f in os.listdir(user_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if images:
                first_image = os.path.join(user_folder, images[0])
    else:
        user_id = None
        result = "failed"
        first_image = './user_images/Unknown.png'

    conn = sqlite3.connect("database/face_lock.db")
    c = conn.cursor()
    c.execute("INSERT INTO access_logs (user_id, result) VALUES (?, ?)", (user_id, result))
    conn.commit()
    conn.close()

    return user_id, result, first_image

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