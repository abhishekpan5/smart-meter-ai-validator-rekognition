import boto3
import io
import re
import time
import sqlite3
from PIL import Image

# AWS configuration
MODEL_ARN = "arn:aws:rekognition:us-east-1:419535873181:project/Smart-meter/version/Smart-meter.2025-06-25T19.03.51/1750858432151"
BUCKET_NAME = "smart-meter-ai-demo"
IMAGE_NAME = "img07.png"  # Change as needed or loop over images
MIN_CONF = 40
DYNAMODB_TABLE = "MeterValidationResults"
SQLITE_DB = "manual_readings.db"

rekognition = boto3.client('rekognition')
s3 = boto3.client('s3')
textract = boto3.client('textract')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)

def detect_custom_labels(bucket, photo, model_arn, min_confidence):
    response = rekognition.detect_custom_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
        MinConfidence=min_confidence,
        ProjectVersionArn=model_arn
    )
    return response

def crop_meter_region(bucket, photo, custom_labels_response, label_name='Meter'):
    # Download image from S3
    s3_response = s3.get_object(Bucket=bucket, Key=photo)
    image_bytes = s3_response['Body'].read()
    image = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = image.size

    # Find the bounding box for the desired label
    for label in custom_labels_response['CustomLabels']:
        if label['Name'].lower() == label_name.lower() and 'Geometry' in label:
            box = label['Geometry']['BoundingBox']
            left = int(img_width * box['Left'])
            top = int(img_height * box['Top'])
            width = int(img_width * box['Width'])
            height = int(img_height * box['Height'])
            right = left + width
            bottom = top + height
            cropped_img = image.crop((left, top, right, bottom))
            return cropped_img
    return None

def extract_text_textract(pil_image):
    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Call Textract to detect text
    response = textract.detect_document_text(Document={'Bytes': img_bytes})

    # Extract lines of text
    lines = []
    for block in response['Blocks']:
        if block['BlockType'] == 'LINE':
            lines.append(block['Text'])
    return lines

def extract_numeric_readings(detected_lines):
    readings = []
    for line in detected_lines:
        # Extract leading numeric part (integer or decimal)
        match = re.match(r'^(\d+\.?\d*)', line.strip())
        if match:
            numeric_value = match.group(1)
            readings.append(numeric_value)
    return readings

def get_manual_reading(image_id):
    """Retrieve manual reading from SQLite database"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT reading FROM manual_readings WHERE image_id = ?", (image_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except sqlite3.Error as e:
        print(f"Database error for {image_id}: {str(e)}")
        return None

def validate_readings(extracted, manual):
    """Compare readings with tolerance"""
    if extracted is None or manual is None:
        return False
    try:
        return abs(float(extracted) - float(manual)) <= max(0.01 * float(manual), 0.1)  # 1% or 0.1 unit tolerance
    except Exception:
        return False

def store_result(image_id, extracted, manual, is_valid):
    """Store validation result in DynamoDB"""
    try:
        table.put_item(
            Item={
                'ImageID': image_id,
                'ExtractedReading': float(extracted) if extracted else None,
                'ManualReading': float(manual) if manual else None,
                'IsValid': is_valid,
                'Timestamp': int(time.time())
            }
        )
    except Exception as e:
        print(f"DynamoDB error for {image_id}: {str(e)}")

def main():
    # Step 1: Detect custom labels
    print(f"Detecting custom labels in {IMAGE_NAME}...")
    custom_labels_response = detect_custom_labels(BUCKET_NAME, IMAGE_NAME, MODEL_ARN, MIN_CONF)
    print("Detected custom labels:")
    for label in custom_labels_response['CustomLabels']:
        print(f"  - {label['Name']} (Confidence: {label['Confidence']:.2f})")

    # Step 2: Crop meter region
    cropped_meter = crop_meter_region(BUCKET_NAME, IMAGE_NAME, custom_labels_response, label_name='Meter')
    if cropped_meter is None:
        print("No meter region detected in the image.")
        return

    # Step 3: Extract text from cropped meter region using Textract
    print("Extracting text from cropped meter region using Textract...")
    detected_lines = extract_text_textract(cropped_meter)
    print("Detected text lines in meter region:")
    for line in detected_lines:
        print(f"  - {line}")

    # Step 4: Extract numeric readings from detected lines
    readings = extract_numeric_readings(detected_lines)
    if readings:
        print("Extracted numeric readings:")
        for reading in readings:
            print(f"  - {reading}")
        extracted_reading = readings[0]
    else:
        print("No numeric readings found in the meter region.")
        extracted_reading = None

    # Step 5: Get manual reading from SQLite
    manual_reading = get_manual_reading(IMAGE_NAME)
    print(f"Manual reading from DB: {manual_reading}")

    # Step 6: Validate and store
    is_valid = validate_readings(extracted_reading, manual_reading)
    print(f"Validation result: {'PASS' if is_valid else 'FAIL'}")
    store_result(IMAGE_NAME, extracted_reading, manual_reading, is_valid)

if __name__ == "__main__":
    main()
