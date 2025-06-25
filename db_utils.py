import sqlite3
from config import SQLITE_DB

def get_manual_reading(image_id):
    """Retrieve manual reading from SQLite database"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT reading FROM manual_readings WHERE image_id = ?", (image_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None  # Return as string to preserve leading zeros
    except sqlite3.Error as e:
        print(f"Database error for {image_id}: {str(e)}")
        return None 