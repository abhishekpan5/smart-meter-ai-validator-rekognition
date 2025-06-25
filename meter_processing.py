import io
import time
from PIL import Image
from config import S3_BUCKET, MODEL_ARN, MIN_CONFIDENCE
from aws_clients import s3, rekognition, table
from db_utils import get_manual_reading

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
        response = rekognition.detect_custom_labels(
            Image={'S3Object': {'Bucket': S3_BUCKET, 'Name': image_key}},
            MinConfidence=MIN_CONFIDENCE,
            ProjectVersionArn=MODEL_ARN
        )
        meter_label = next((label for label in response['CustomLabels'] 
                          if label['Name'] == 'Meter' and 'Geometry' in label), None)
        if not meter_label:
            print(f"No meter detected in {image_key}")
            return None, None
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
        cropped_img = image.crop((left, top, left + width, top + height))
        img_byte_arr = io.BytesIO()
        cropped_img.save(img_byte_arr, format='JPEG')
        text_response = rekognition.detect_text(
            Image={'Bytes': img_byte_arr.getvalue()}
        )
        readings = [text['DetectedText'] 
                   for text in text_response['TextDetections'] 
                   if text['Type'] == 'LINE' 
                   and text['DetectedText'].replace('.', '').isdigit()]
        return float(readings[0]) if readings else None, meter_label['Confidence']
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