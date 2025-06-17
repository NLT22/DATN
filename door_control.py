from queue import Queue
import threading

door_status = "CLOSED"
client_queues = []  # danh sách chứa các hàng đợi của từng client
auto_close_timer = None
door_lock = threading.Lock()

def set_door_status(status):
    global door_status
    with door_lock:
        if status != door_status:
            door_status = status
            for q in client_queues:
                q.put(status)  # phát cho tất cả hàng đợi của client
            print(f"[INFO] Door set to: {status}")

def get_door_status():
    with door_lock:
        return door_status

def register_client_queue():
    q = Queue()
    client_queues.append(q)
    return q

def unregister_client_queue(q):
    if q in client_queues:
        client_queues.remove(q)

def auto_close_door(delay=1.0):
    global auto_close_timer
    if auto_close_timer:
        auto_close_timer.cancel()

    def close():
        set_door_status("CLOSED")
        print("🔒 Door auto-closed")

    auto_close_timer = threading.Timer(delay, close)
    auto_close_timer.start()
