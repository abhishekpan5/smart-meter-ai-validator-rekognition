import sqlite3
import pandas as pd
import os

def create_database():
    """Create SQLite database and load CSV data"""
    
    # Database file name
    DB_FILE = 'manual_readings.db'
    
    # Remove existing database if it exists
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Removed existing database: {DB_FILE}")
    
    # Create connection
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Create table with TEXT type to preserve leading zeros
        cursor.execute('''
            CREATE TABLE manual_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL UNIQUE,
                reading TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        print("Created table: manual_readings")
        
        # Read CSV file with meter_reading as string to preserve leading zeros
        df = pd.read_csv('ref_meter_data.csv', dtype={'meter_reading': str})
        print(f"Loaded CSV data: {len(df)} records")
        
        # Insert data into database
        for index, row in df.iterrows():
            image_id = row['imageid']
            meter_reading = row['meter_reading']  # Already string, preserves leading zeros
            
            cursor.execute('''
                INSERT INTO manual_readings (image_id, reading)
                VALUES (?, ?)
            ''', (image_id, meter_reading))
            
            print(f"Inserted: {image_id} -> {meter_reading}")
        
        # Commit changes
        conn.commit()
        print(f"\nSuccessfully loaded {len(df)} records into database")
        
        # Verify data
        print("\nVerifying data in database:")
        cursor.execute('SELECT image_id, reading FROM manual_readings ORDER BY image_id')
        results = cursor.fetchall()
        
        for image_id, reading in results:
            print(f"  {image_id}: {reading} (length: {len(reading)})")
        
        print(f"\nDatabase created successfully: {DB_FILE}")
        
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        conn.rollback()
        raise
    finally:
        conn.close()

def verify_data():
    """Verify that leading zeros are preserved"""
    conn = sqlite3.connect('manual_readings.db')
    cursor = conn.cursor()
    
    try:
        # Check original CSV data
        df = pd.read_csv('ref_meter_data.csv', dtype={'meter_reading': str})
        print("Original CSV data:")
        for index, row in df.iterrows():
            reading = row['meter_reading']  # Already string
            print(f"  {row['imageid']}: {reading} (length: {len(reading)})")
        
        print("\nDatabase data:")
        cursor.execute('SELECT image_id, reading FROM manual_readings ORDER BY image_id')
        results = cursor.fetchall()
        
        for image_id, reading in results:
            print(f"  {image_id}: {reading} (length: {len(reading)})")
        
        # Verify they match
        print("\nVerification:")
        csv_data = dict(zip(df['imageid'], df['meter_reading']))
        
        for image_id, reading in results:
            csv_reading = csv_data.get(image_id, '')
            if reading == csv_reading:
                print(f"  ✅ {image_id}: MATCH")
            else:
                print(f"  ❌ {image_id}: MISMATCH - CSV: {csv_reading}, DB: {reading}")
                
    except Exception as e:
        print(f"Error verifying data: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("Setting up SQLite database...")
    create_database()
    
    print("\n" + "="*50)
    print("Verifying data integrity...")
    verify_data()
    
    print("\nDatabase setup complete!") 