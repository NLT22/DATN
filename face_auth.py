# face_auth.py
import cv2
from anti_spoof import AntiSpoof

spoof_model = AntiSpoof(
    weights="./models/AntiSpoofing_bin_1.5_128.onnx",
    model_img_size=128,
    force_cpu=False
)

def is_real_face(frame, return_score=False):
    preds = spoof_model([frame])  # Trả về list [[fake_prob, real_prob]]
    if preds is None or len(preds) == 0:
        return False if not return_score else [[0.0, 0.0]]
    return preds[0][1] > 0.5 if not return_score else preds