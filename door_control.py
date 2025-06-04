from queue import Queue
import threading

door_status = "CLOSED"
door_event_queue = Queue()
auto_close_timer = None

def set_door_status(status):
    global door_status
    if status != door_status:
        door_status = status
        door_event_queue.put(status)  # báo SSE cập nhật

def get_door_status():
    return door_status

def get_door_event_queue():
    return door_event_queue

def auto_close_door(delay=2.0):
    global auto_close_timer
    if auto_close_timer:
        auto_close_timer.cancel()

    def close():
        set_door_status("CLOSED")
        print("🔒 Door auto-closed")

    auto_close_timer = threading.Timer(delay, close)
    auto_close_timer.start()
