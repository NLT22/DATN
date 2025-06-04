# face_utils.py
import faiss
import numpy as np
import pickle
import os
import sqlite3

FAISS_INDEX_PATH = "faiss_index/index.bin"
ID_MAPPING_PATH = "embeddings/id_to_user.pkl"

def create_faiss_index(dim=512):
    index = faiss.IndexFlatL2(dim)  # dùng L2 distance
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
