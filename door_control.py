# door_control.py

import threading

door_status = "CLOSED"
auto_close_timer = None  # Dùng để reset timeout khi phát hiện liên tục

def set_door_status(status):
    global door_status
    door_status = status

def get_door_status():
    return door_status

def auto_close_door(delay=2.0):
    global auto_close_timer

    # Nếu có timer cũ -> hủy
    if auto_close_timer is not None:
        auto_close_timer.cancel()

    # Tạo timer mới để đóng cửa
    def close():
        set_door_status("CLOSED")
        print("🔒 Door auto-closed")

    auto_close_timer = threading.Timer(delay, close)
    auto_close_timer.start()
