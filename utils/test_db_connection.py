import sqlite3

def get_manual_reading(image_id):
    """Retrieve manual reading from SQLite database"""
    try:
        conn = sqlite3.connect('manual_readings.db')
        cursor = conn.cursor()
        cursor.execute("SELECT reading FROM manual_readings WHERE image_id = ?", (image_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None  # Return as string to preserve leading zeros
    except sqlite3.Error as e:
        print(f"Database error for {image_id}: {str(e)}")
        return None

def validate_readings(extracted, manual):
    """Compare readings with tolerance"""
    if extracted is None or manual is None:
        return False
    
    # Convert both to float for numerical comparison
    try:
        extracted_float = float(extracted)
        manual_float = float(manual)
        return abs(extracted_float - manual_float) <= max(0.01 * manual_float, 0.1)  # 1% or 0.1 unit tolerance
    except ValueError:
        # If conversion fails, do exact string comparison
        return str(extracted).strip() == str(manual).strip()

def test_database_connection():
    """Test that database functions work properly"""
    
    print("Testing database connection and reading functions...")
    
    # Test reading each image ID
    test_image_ids = ['img01', 'img02', 'img03', 'img04', 'img05', 'img06', 'img07']
    
    for image_id in test_image_ids:
        reading = get_manual_reading(image_id)
        print(f"  {image_id}: {reading} (type: {type(reading)})")
    
    print("\nTesting validation function with leading zeros:")
    
    # Test validation with leading zeros
    test_cases = [
        ('img01', '039672', '039672'),  # Exact match
        ('img02', '014673', '014673'),  # Exact match
        ('img03', '021043', '021043'),  # Exact match
        ('img01', '39672', '039672'),   # Missing leading zero
        ('img02', '14673', '014673'),   # Missing leading zero
    ]
    
    for image_id, extracted, expected_manual in test_cases:
        manual = get_manual_reading(image_id)
        is_valid = validate_readings(extracted, manual)
        print(f"  Extracted: {extracted} | Manual: {manual} | Expected: {expected_manual} | Valid: {is_valid}")

if __name__ == "__main__":
    test_database_connection() 