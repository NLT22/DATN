import sqlite3

def view_table_contents(db_path="database/face_lock.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    print("\n📋 USERS TABLE:")
    c.execute("SELECT * FROM users")
    users = c.fetchall()
    for row in users:
        print(row)

    print("\n📋 ACCESS_LOGS TABLE:")
    c.execute("SELECT * FROM access_logs")
    logs = c.fetchall()
    for row in logs:
        print(row)

    print("\n📋 USER_IMAGES TABLE:")
    c.execute("SELECT * FROM user_images")
    images = c.fetchall()
    for row in images:
        print(row)

    conn.close()

if __name__ == "__main__":
    view_table_contents()

