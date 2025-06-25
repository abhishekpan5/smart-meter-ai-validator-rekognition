from meter_processing import start_model, extract_meter_data, validate_readings, store_result
from aws_clients import s3
from config import S3_BUCKET
from db_utils import get_manual_reading

def process_batch_images():
    start_model()
    bucket = s3.Bucket(S3_BUCKET)
    images = [obj.key for obj in bucket.objects.all() if obj.key.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} images to process")
    for image_key in images:
        print(f"\nProcessing {image_key}...")
        extracted_reading, confidence = extract_meter_data(image_key)
        if extracted_reading is None:
            print("Skipping - No reading extracted")
            continue
        manual_reading = get_manual_reading(image_key)
        if manual_reading is None:
            print("Skipping - No manual reading found")
            continue
        is_valid = validate_readings(extracted_reading, manual_reading)
        store_result(image_key, extracted_reading, confidence, manual_reading, is_valid)
        extracted_str = f"{extracted_reading:.2f}" if isinstance(extracted_reading, float) else str(extracted_reading)
        manual_str = f"{manual_reading:.2f}" if isinstance(manual_reading, float) else str(manual_reading)
        print(f"Extracted: {extracted_str} | Manual: {manual_str} | Validation: {'PASS' if is_valid else 'FAIL'}")

if __name__ == "__main__":
    process_batch_images()
