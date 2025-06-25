import boto3
import io
from PIL import Image

# AWS configuration
MODEL_ARN = "arn:aws:rekognition:us-east-1:419535873181:project/Smart-meter/version/Smart-meter.2025-06-25T19.03.51/1750858432151"
BUCKET_NAME = "smart-meter-ai-demo"
IMAGE_NAME = "img07.png"  # Change as needed or loop over images
MIN_CONF = 40

rekognition = boto3.client('rekognition')
s3 = boto3.client('s3')
textract = boto3.client('textract')

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

if __name__ == "__main__":
    main()
