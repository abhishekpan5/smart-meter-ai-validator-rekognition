import io
import time
from PIL import Image
from config import S3_BUCKET, MODEL_ARN, MIN_CONFIDENCE
from aws_clients import s3, rekognition, table
from db_utils import get_manual_reading

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

def extract_meter_data(image_key):
    """Extract meter reading using custom model and OCR"""
    try:
        # Detect custom labels
        response = rekognition.detect_custom_labels(
            Image={'S3Object': {'Bucket': S3_BUCKET, 'Name': image_key}},
            MinConfidence=MIN_CONFIDENCE,
            ProjectVersionArn=MODEL_ARN
        )
        
        # Validate response
        if 'CustomLabels' not in response:
            print(f"No custom labels found in {image_key}")
            return None, None
        
        # Debug: Print all detected custom labels
        print(f"Detected custom labels in {image_key}:")
        for label in response['CustomLabels']:
            print(f"  - {label.get('Name')}: {label.get('Confidence', 0):.2f}%")
        
        # Find meter boundary
        meter_label = None
        for label in response['CustomLabels']:
            if label.get('Name') == 'Meter Reading' and 'Geometry' in label:
                meter_label = label
                break
        
        if not meter_label:
            print(f"No meter detected in {image_key}")
            return None, None
        
        # Get image and crop meter region
        try:
            s3_object = s3.Object(S3_BUCKET, image_key)
            s3_response = s3_object.get()
            stream = io.BytesIO(s3_response['Body'].read())
            image = Image.open(stream)
        except Exception as e:
            print(f"Error downloading image {image_key}: {str(e)}")
            return None, None
        
        # Calculate crop coordinates
        img_width, img_height = image.size
        box = meter_label['Geometry']['BoundingBox']
        
        # Validate bounding box coordinates
        left = max(0, int(img_width * box['Left']))
        top = max(0, int(img_height * box['Top']))
        width = min(int(img_width * box['Width']), img_width - left)
        height = min(int(img_height * box['Height']), img_height - top)
        
        # Ensure valid crop dimensions
        if width <= 0 or height <= 0:
            print(f"Invalid crop dimensions for {image_key}")
            return None, None
        
        # Crop and process meter region
        cropped_img = image.crop((left, top, left + width, top + height))
        
        # Convert to bytes for text detection
        img_byte_arr = io.BytesIO()
        cropped_img.save(img_byte_arr, format='JPEG', quality=95)
        img_bytes = img_byte_arr.getvalue()
        
        # Validate image size for Rekognition (max 5MB)
        if len(img_bytes) > 5 * 1024 * 1024:
            print(f"Image too large for text detection: {len(img_bytes)} bytes")
            return None, None
        
        # Detect text in cropped image
        text_response = rekognition.detect_text(
            Image={'Bytes': img_bytes}
        )
        
        # Validate text response
        if 'TextDetections' not in text_response:
            print(f"No text detected in {image_key}")
            return None, None
        
        # Find numeric readings (handles decimal values)
        readings = []
        for text in text_response['TextDetections']:
            if text.get('Type') == 'LINE':
                detected_text = text.get('DetectedText', '')
                # Check if text contains only digits and decimal points
                if detected_text.replace('.', '').replace(',', '').isdigit():
                    readings.append(detected_text)
        
        if not readings:
            print(f"No numeric readings found in {image_key}")
            return None, None
        
        # Return first reading and confidence
        try:
            reading_value = float(readings[0].replace(',', ''))
            confidence = meter_label.get('Confidence', 0.0)
            return reading_value, confidence
        except ValueError:
            print(f"Invalid reading format: {readings[0]}")
            return None, None
            
    except Exception as e:
        print(f"Error processing {image_key}: {str(e)}")
        return None, None

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