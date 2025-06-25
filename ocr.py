import boto3
import io
import time
import sqlite3
from PIL import Image
from config import S3_BUCKET, MODEL_ARN, MIN_CONFIDENCE, DYNAMODB_TABLE, SQLITE_DB

# AWS configuration
rekognition = boto3.client('rekognition')
s3 = boto3.client('s3')
textract = boto3.client('textract')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)

def start_model():
    """Ensure custom model is running"""
    try:
        # Check current model status
        response = rekognition.describe_project_versions(
            ProjectVersionArns=[MODEL_ARN]
        )
        
        if not response.get('ProjectVersionDescriptions'):
            raise Exception(f"Model {MODEL_ARN} not found")
        
        status = response['ProjectVersionDescriptions'][0]['Status']
        
        if status == 'RUNNING':
            print("Model already RUNNING")
            return
        
        if status == 'STOPPING':
            print("Model is stopping, waiting...")
            time.sleep(30)
        
        # Start the model
        print("Starting model...")
        rekognition.start_project_version(
            ProjectVersionArn=MODEL_ARN,
            MinInferenceUnits=1
        )
        
        # Wait for model to be running with exponential backoff
        max_attempts = 30  # 5 minutes max
        attempt = 0
        while attempt < max_attempts:
            try:
                response = rekognition.describe_project_versions(
                    ProjectVersionArns=[MODEL_ARN]
                )
                current_status = response['ProjectVersionDescriptions'][0]['Status']
                
                if current_status == 'RUNNING':
                    print("Model is RUNNING")
                    return
                elif current_status == 'FAILED':
                    raise Exception(f"Model failed to start: {current_status}")
                elif current_status == 'STOPPING':
                    print("Model is stopping, waiting...")
                    time.sleep(30)
                else:
                    print(f"Model status: {current_status}")
                    time.sleep(min(10 * (2 ** attempt), 60))  # Exponential backoff, max 60s
                    attempt += 1
                    
            except Exception as e:
                print(f"Error checking model status: {str(e)}")
                time.sleep(10)
                attempt += 1
        
        raise Exception("Model failed to start within timeout period")
        
    except Exception as e:
        print(f"Model start error: {str(e)}")
        raise

def detect_custom_labels(bucket, photo, model_arn, min_confidence):
    """Detect custom labels using Rekognition"""
    try:
        response = rekognition.detect_custom_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
            MinConfidence=min_confidence,
            ProjectVersionArn=model_arn
        )
        return response
    except Exception as e:
        print(f"Error detecting custom labels: {str(e)}")
        return None

def crop_meter_region(bucket, photo, custom_labels_response, label_name='Meter Reading'):
    """Crop meter region based on custom label detection"""
    try:
        # Download image from S3
        s3_response = s3.get_object(Bucket=bucket, Key=photo)
        image_bytes = s3_response['Body'].read()
        image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = image.size

        # Find the bounding box for the desired label
        for label in custom_labels_response['CustomLabels']:
            if label['Name'].lower() == label_name.lower() and 'Geometry' in label:
                box = label['Geometry']['BoundingBox']
                
                # Validate bounding box coordinates
                left = max(0, int(img_width * box['Left']))
                top = max(0, int(img_height * box['Top']))
                width = min(int(img_width * box['Width']), img_width - left)
                height = min(int(img_height * box['Height']), img_height - top)
                
                # Ensure valid crop dimensions
                if width <= 0 or height <= 0:
                    print(f"Invalid crop dimensions for {photo}")
                    return None, None
                
                right = left + width
                bottom = top + height
                cropped_img = image.crop((left, top, right, bottom))
                return cropped_img, label.get('Confidence', 0.0)
        
        print(f"No {label_name} label found in {photo}")
        return None, None
        
    except Exception as e:
        print(f"Error cropping meter region: {str(e)}")
        return None, None

def extract_text_textract(pil_image):
    """Extract text using AWS Textract"""
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Validate image size for Textract (max 5MB)
        if len(img_bytes) > 5 * 1024 * 1024:
            print(f"Image too large for Textract: {len(img_bytes)} bytes")
            return []

        # Call Textract to detect text
        response = textract.detect_document_text(Document={'Bytes': img_bytes})

        # Extract lines of text
        lines = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                lines.append(block['Text'])
        return lines
        
    except Exception as e:
        print(f"Error extracting text with Textract: {str(e)}")
        return []

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

def validate_readings(extracted, manual):
    """Compare readings with tolerance"""
    if extracted is None or manual is None:
        return False
    try:
        extracted_float = float(extracted)
        manual_float = float(manual)
        return abs(extracted_float - manual_float) <= max(0.01 * manual_float, 0.1)
    except ValueError:
        return str(extracted).strip() == str(manual).strip()

def store_result(image_id, extracted, confidence, manual, is_valid, ocr_method="Textract"):
    """Store validation result in DynamoDB"""
    try:
        table.put_item(
            Item={
                'ImageID': image_id,
                'ExtractedReading': extracted,
                'Confidence': confidence,
                'ManualReading': manual,
                'IsValid': is_valid,
                'OCRMethod': ocr_method,  # Track which OCR method was used
                'Timestamp': int(time.time())
            }
        )
        print(f"Result stored in DynamoDB for {image_id}")
    except Exception as e:
        print(f"DynamoDB error for {image_id}: {str(e)}")

def extract_meter_data_textract(image_key):
    """Extract meter reading using custom model and Textract OCR"""
    try:
        # Step 1: Detect custom labels
        print(f"Detecting custom labels in {image_key}...")
        custom_labels_response = detect_custom_labels(S3_BUCKET, image_key, MODEL_ARN, MIN_CONFIDENCE)
        
        if not custom_labels_response or 'CustomLabels' not in custom_labels_response:
            print(f"No custom labels found in {image_key}")
            return None, None
        
        # Debug: Print all detected custom labels
        print(f"Detected custom labels in {image_key}:")
        for label in custom_labels_response['CustomLabels']:
            print(f"  - {label.get('Name')}: {label.get('Confidence', 0):.2f}%")
        
        # Step 2: Crop meter region
        cropped_meter, confidence = crop_meter_region(S3_BUCKET, image_key, custom_labels_response, 'Meter Reading')
        if cropped_meter is None:
            print(f"No meter region detected in {image_key}")
            return None, None
        
        # Step 3: Extract text from cropped meter region using Textract
        print(f"Extracting text from cropped meter region using Textract...")
        detected_lines = extract_text_textract(cropped_meter)
        
        if not detected_lines:
            print(f"No text detected in {image_key}")
            return None, None
        
        print(f"Detected text lines in meter region:")
        for line in detected_lines:
            print(f"  - {line}")
        
        # Step 4: Find numeric readings
        readings = []
        for line in detected_lines:
            # Clean the text and check if it's numeric
            cleaned_text = line.replace(',', '').replace(' ', '')
            if cleaned_text.replace('.', '').isdigit():
                readings.append(line)
        
        if not readings:
            print(f"No numeric readings found in {image_key}")
            return None, None
        
        # Return first reading and confidence
        try:
            reading_value = float(readings[0].replace(',', ''))
            return reading_value, confidence
        except ValueError:
            print(f"Invalid reading format: {readings[0]}")
            return None, None
            
    except Exception as e:
        print(f"Error processing {image_key}: {str(e)}")
        return None, None

def process_batch_images_textract():
    """Process all images in S3 bucket using Textract OCR"""
    start_model()
    
    # List images in bucket
    try:
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(S3_BUCKET)
        images = [obj.key for obj in bucket.objects.all() if obj.key.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        print(f"Error listing images in bucket: {str(e)}")
        return
    
    print(f"Found {len(images)} images to process with Textract")
    
    for image_key in images:
        print(f"\nProcessing {image_key} with Textract...")
        
        # Extract reading from image using Textract
        extracted_reading, confidence = extract_meter_data_textract(image_key)
        
        if extracted_reading is None:
            print("Skipping - No reading extracted")
            continue
        
        # Get manual reading
        manual_reading = get_manual_reading(image_key)
        
        if manual_reading is None:
            print("Skipping - No manual reading found")
            continue
        
        # Validate and store result
        is_valid = validate_readings(extracted_reading, manual_reading)
        store_result(image_key, extracted_reading, confidence, manual_reading, is_valid, "Textract")
        
        # Format output
        extracted_str = f"{extracted_reading:.2f}" if isinstance(extracted_reading, float) else str(extracted_reading)
        manual_str = f"{manual_reading:.2f}" if isinstance(manual_reading, float) else str(manual_reading)
        print(f"Extracted: {extracted_str} | Manual: {manual_str} | Validation: {'PASS' if is_valid else 'FAIL'}")

def main():
    """Main function for single image processing"""
    IMAGE_NAME = "img07.png"  # Change as needed
    
    print("=== Textract OCR Processing ===")
    print(f"Processing single image: {IMAGE_NAME}")
    
    # Process single image
    extracted_reading, confidence = extract_meter_data_textract(IMAGE_NAME)
    
    if extracted_reading is not None:
        # Get manual reading
        manual_reading = get_manual_reading(IMAGE_NAME)
        
        if manual_reading is not None:
            # Validate and store result
            is_valid = validate_readings(extracted_reading, manual_reading)
            store_result(IMAGE_NAME, extracted_reading, confidence, manual_reading, is_valid, "Textract")
            
            # Format output
            extracted_str = f"{extracted_reading:.2f}" if isinstance(extracted_reading, float) else str(extracted_reading)
            manual_str = f"{manual_reading:.2f}" if isinstance(manual_reading, float) else str(manual_reading)
            print(f"\nFinal Result:")
            print(f"Extracted: {extracted_str} | Manual: {manual_str} | Validation: {'PASS' if is_valid else 'FAIL'}")
        else:
            print("No manual reading found in database")
    else:
        print("Failed to extract reading from image")

if __name__ == "__main__":
    # Uncomment one of the following lines:
    main()  # Process single image
    # process_batch_images_textract()  # Process all images in bucket
