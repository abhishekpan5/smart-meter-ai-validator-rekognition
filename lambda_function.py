import boto3
import json
import sqlite3
import io
import time
from PIL import Image, ImageDraw

# Configuration
S3_BUCKET = 'your-image-bucket'
MODEL_ARN = 'arn:aws:rekognition:us-east-1:419535873181:project/Smart-meter/version/Smart-meter.2025-06-25T19.03.51/1750858432151'
DYNAMODB_TABLE = 'MeterValidationResults'
MIN_CONFIDENCE = 95
SQLITE_DB = 'manual_readings.db'

# Initialize AWS clients
s3 = boto3.resource('s3')
rekognition = boto3.client('rekognition')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)

def start_model():
    """Ensure custom model is running"""
    try:
        status = rekognition.describe_project_versions(
            ProjectVersionArns=[MODEL_ARN]
        )['ProjectVersionDescriptions'][0]['Status']
        
        if status != 'RUNNING':
            rekognition.start_project_version(
                ProjectVersionArn=MODEL_ARN,
                MinInferenceUnits=1
            )
            print("Starting model...")
            while True:
                current_status = rekognition.describe_project_versions(
                    ProjectVersionArns=[MODEL_ARN]
                )['ProjectVersionDescriptions'][0]['Status']
                if current_status == 'RUNNING':
                    print("Model is RUNNING")
                    break
                time.sleep(10)
        else:
            print("Model already RUNNING")
    except Exception as e:
        print(f"Model start error: {str(e)}")
        raise

def extract_meter_data(image_key):
    """Extract meter reading using custom model and OCR"""
    try:
        # Detect custom labels
        response = rekognition.detect_custom_labels(
            Image={'S3Object': {'Bucket': S3_BUCKET, 'Name': image_key}},
            MinConfidence=MIN_CONFIDENCE,
            ProjectVersionArn=MODEL_ARN
        )
        
        # Find meter boundary
        meter_label = next((label for label in response['CustomLabels'] 
                          if label['Name'] == 'Meter' and 'Geometry' in label), None)
        
        if not meter_label:
            print(f"No meter detected in {image_key}")
            return None, None
        
        # Get image and crop meter region
        s3_object = s3.Object(S3_BUCKET, image_key)
        s3_response = s3_object.get()
        stream = io.BytesIO(s3_response['Body'].read())
        image = Image.open(stream)
        
        img_width, img_height = image.size
        box = meter_label['Geometry']['BoundingBox']
        left = img_width * box['Left']
        top = img_height * box['Top']
        width = img_width * box['Width']
        height = img_height * box['Height']
        
        # Crop and process meter region
        cropped_img = image.crop((left, top, left + width, top + height))
        
        # Convert to bytes for text detection
        img_byte_arr = io.BytesIO()
        cropped_img.save(img_byte_arr, format='JPEG')
        
        # Detect text in cropped image
        text_response = rekognition.detect_text(
            Image={'Bytes': img_byte_arr.getvalue()}
        )
        
        # Find numeric readings (handles decimal values)
        readings = [text['DetectedText'] 
                   for text in text_response['TextDetections'] 
                   if text['Type'] == 'LINE' 
                   and text['DetectedText'].replace('.', '').isdigit()]
        
        return float(readings[0]) if readings else None, meter_label['Confidence']
        
    except Exception as e:
        print(f"Error processing {image_key}: {str(e)}")
        return None, None

def get_manual_reading(image_id):
    """Retrieve manual reading from SQLite database"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT reading FROM manual_readings WHERE image_id = ?", (image_id,))
        result = cursor.fetchone()
        conn.close()
        return float(result[0]) if result else None
    except sqlite3.Error as e:
        print(f"Database error for {image_id}: {str(e)}")
        return None

def validate_readings(extracted, manual):
    """Compare readings with tolerance"""
    if extracted is None or manual is None:
        return False
    return abs(extracted - manual) <= max(0.01 * manual, 0.1)  # 1% or 0.1 unit tolerance

def store_result(image_id, extracted, confidence, manual, is_valid):
    """Store validation result in DynamoDB"""
    try:
        table.put_item(
            Item={
                'ImageID': image_id,
                'ExtractedReading': extracted,
                'Confidence': confidence,
                'ManualReading': manual,
                'IsValid': is_valid,
                'Timestamp': int(time.time())
            }
        )
    except Exception as e:
        print(f"DynamoDB error for {image_id}: {str(e)}")

def lambda_handler(event, context):
    """Lambda function handler triggered by S3 uploads"""
    try:
        # Extract image key from S3 event
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            image_key = record['s3']['object']['key']
            
            # Only process image files
            if not image_key.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {image_key}")
                continue
            
            print(f"Processing uploaded image: {image_key}")
            
            # Ensure model is running
            start_model()
            
            # Extract reading from image
            extracted_reading, confidence = extract_meter_data(image_key)
            
            if extracted_reading is None:
                print("No reading extracted")
                continue
                
            # Get manual reading
            manual_reading = get_manual_reading(image_key)
            
            if manual_reading is None:
                print("No manual reading found")
                continue
                
            # Validate and store result
            is_valid = validate_readings(extracted_reading, manual_reading)
            store_result(image_key, extracted_reading, confidence, manual_reading, is_valid)
            
            print(f"Extracted: {extracted_reading:.2f} | Manual: {manual_reading:.2f} | Validation: {'PASS' if is_valid else 'FAIL'}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Processing completed successfully')
        }
        
    except Exception as e:
        print(f"Lambda execution error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        } 