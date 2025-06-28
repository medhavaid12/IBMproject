
import sqlite3
from datetime import datetime
from collections import Counter


DB_FILE = "mask_violations.db"

def analyze_violation_timestamps(db_file):
    """
    Connects to the SQLite database, fetches all violation records,
    and analyzes the timestamps to provide daily statistics.
    """
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Query all records from the violations table
        cursor.execute("SELECT label, timestamp FROM violations ORDER BY timestamp")
        records = cursor.fetchall()

        conn.close() # Close the database connection

        if not records:
            print(f"😔 No violation records found in {db_file}.")
            return

        print(f"--- Mask Violation Statistics from {db_file} ---")
        print(f"Total violations logged: {len(records)}")
        print("\nViolations by Label:")
        label_counts = Counter(record[0] for record in records)
        for label, count in label_counts.items():
            print(f"  - {label}: {count} incidents")

        print("\nViolations by Date:")
        # Extract just the date part from the timestamp (YYYY-MM-DD)
        dates = [datetime.fromisoformat(record[1]).strftime('%Y-%m-%d') for record in records]
        date_counts = Counter(dates)

        # Sort dates for chronological display
        sorted_dates = sorted(date_counts.keys())

        for date in sorted_dates:
            print(f"  - {date}: {date_counts[date]} violations")

        print("\n--- End of Statistics ---")

    except sqlite3.OperationalError as e:
        print(f"❌ Database Error: {e}")
        print(f"Please ensure '{db_file}' exists and is not locked by another process (like the running Flask app).")
        print("You might need to stop the Flask app before running this analysis script.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# ======================= Run Analysis =======================
if __name__ == '__main__':
    analyze_violation_timestamps(DB_FILE)
import sqlite3
from datetime import datetime
from collections import Counter

# ======================= CONFIG =======================
# Path to your SQLite database file
# Make sure this path is correct relative to where you run this script.
# It should be in the same directory as your app.py file.
DB_FILE = "mask_violations.db"

# ======================= Analyze Violations =======================
def analyze_violation_timestamps(db_file):
    """
    Connects to the SQLite database, fetches all violation records,
    and analyzes the timestamps to provide daily statistics.
    """
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Query all records from the violations table
        cursor.execute("SELECT label, timestamp FROM violations ORDER BY timestamp")
        records = cursor.fetchall()

        conn.close() # Close the database connection

        if not records:
            print(f"😔 No violation records found in {db_file}.")
            return

        print(f"--- Mask Violation Statistics from {db_file} ---")
        print(f"Total violations logged: {len(records)}")
        print("\nViolations by Label:")
        label_counts = Counter(record[0] for record in records)
        for label, count in label_counts.items():
            print(f"  - {label}: {count} incidents")

        print("\nViolations by Date:")
        # Extract just the date part from the timestamp (YYYY-MM-DD)
        dates = [datetime.fromisoformat(record[1]).strftime('%Y-%m-%d') for record in records]
        date_counts = Counter(dates)

        # Sort dates for chronological display
        sorted_dates = sorted(date_counts.keys())

        for date in sorted_dates:
            print(f"  - {date}: {date_counts[date]} violations")

        print("\n--- End of Statistics ---")

    except sqlite3.OperationalError as e:
        print(f"❌ Database Error: {e}")
        print(f"Please ensure '{db_file}' exists and is not locked by another process (like the running Flask app).")
        print("You might need to stop the Flask app before running this analysis script.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# ======================= Run Analysis =======================
if __name__ == '__main__':
    analyze_violation_timestamps(DB_FILE)

