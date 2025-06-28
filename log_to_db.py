import sqlite3

DB_FILE = "mask_violations.db"
LOG_FILE = "violations.log"

def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT UNIQUE
        )
    """)
    conn.commit()
    return conn

def insert_log_entries():
    conn = setup_database()
    cursor = conn.cursor()

    with open(LOG_FILE, "r") as f:
        for line in f:
            timestamp = line.split(" - ")[0].strip()
            try:
                cursor.execute("INSERT INTO violations (timestamp) VALUES (?)", (timestamp,))
                conn.commit()
            except sqlite3.IntegrityError:
                continue

    conn.close()

if __name__ == "__main__":
    insert_log_entries()
    print("Database created successfully ✅")
