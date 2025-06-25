# Smart Meter Reading Validation System - Execution Flow

## Overview
This document details the complete execution flow of the Smart Meter Reading Validation System, from image upload to result storage.

## System Architecture
```
S3 Bucket (Images) → AWS Rekognition (AI Model) → SQLite DB (Manual Readings) → DynamoDB (Results)
```

## Detailed Execution Flow

### 1. **System Initialization** (`main.py`)
```
Entry Point: python main.py
├── Import all required modules
├── Call process_batch_images()
└── Begin batch processing
```

### 2. **Model Startup** (`meter_processing.py` → `start_model()`)
```
Function: start_model()
├── Check current model status via AWS Rekognition API
├── If model is NOT RUNNING:
│   ├── Start the custom model (MODEL_ARN)
│   ├── Wait for model to reach RUNNING state
│   └── Poll status every 10 seconds
└── If model is RUNNING:
    └── Continue to next step
```

### 3. **Image Discovery** (`main.py` → `process_batch_images()`)
```
Function: process_batch_images()
├── Connect to S3 bucket: 'smart-meter-ai-demo'
├── List all objects in bucket
├── Filter for image files (.png, .jpg, .jpeg)
└── Create list of images to process
```

### 4. **Individual Image Processing Loop**
For each image in the bucket:

#### 4.1 **Meter Detection** (`meter_processing.py` → `extract_meter_data()`)
```
Function: extract_meter_data(image_key)
├── Call AWS Rekognition Custom Labels API
│   ├── Input: S3 image object
│   ├── Model: Custom trained meter detection model
│   └── Confidence threshold: 95%
├── Extract meter boundary coordinates
├── If no meter detected:
│   └── Return (None, None) → Skip image
└── If meter detected:
    └── Continue to OCR processing
```

#### 4.2 **Image Cropping and OCR**
```
├── Download image from S3
├── Crop image to meter boundary
├── Convert cropped image to bytes
├── Call AWS Rekognition Text Detection API
├── Extract numeric readings from detected text
└── Return (extracted_reading, confidence_score)
```

#### 4.3 **Manual Reading Retrieval** (`db_utils.py` → `get_manual_reading()`)
```
Function: get_manual_reading(image_id)
├── Connect to SQLite database: 'manual_readings.db'
├── Query: SELECT reading FROM manual_readings WHERE image_id = ?
├── If reading found:
│   └── Return reading (preserves leading zeros as string)
└── If no reading found:
    └── Return None → Skip image
```

#### 4.4 **Reading Validation** (`meter_processing.py` → `validate_readings()`)
```
Function: validate_readings(extracted, manual)
├── If either reading is None:
│   └── Return False
├── Try numerical comparison:
│   ├── Convert both to float
│   ├── Calculate absolute difference
│   ├── Apply tolerance: max(1% of manual, 0.1 units)
│   └── Return True if within tolerance
└── If conversion fails:
    └── Do exact string comparison
```

#### 4.5 **Result Storage** (`meter_processing.py` → `store_result()`)
```
Function: store_result(image_id, extracted, confidence, manual, is_valid)
├── Connect to DynamoDB table: 'MeterValidationResults'
├── Create item with:
│   ├── ImageID: image_id
│   ├── ExtractedReading: extracted_reading
│   ├── Confidence: confidence_score
│   ├── ManualReading: manual_reading
│   ├── IsValid: validation_result
│   └── Timestamp: current_unix_timestamp
└── Store in DynamoDB
```

#### 4.6 **Output Display**
```
Print formatted results:
├── Extracted reading (formatted)
├── Manual reading (formatted)
└── Validation status (PASS/FAIL)
```

## Data Flow Diagram

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   S3 Bucket │───▶│ AWS Rekognition│───▶│ SQLite DB   │───▶│ DynamoDB    │
│  (Images)   │    │  (AI Model)   │    │(Manual Data)│    │ (Results)   │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│ Image Files │    │ Meter Detect │    │ Manual Read │    │ Validation  │
│ (PNG/JPG)   │    │ + OCR Text   │    │ (with zeros)│    │ Results     │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
```

## Error Handling Flow

### 1. **Model Startup Errors**
- **Issue**: Model fails to start
- **Action**: Raise exception, stop processing
- **Recovery**: Manual intervention required

### 2. **Image Processing Errors**
- **Issue**: No meter detected or OCR fails
- **Action**: Skip image, continue with next
- **Recovery**: Automatic (no manual intervention)

### 3. **Database Errors**
- **Issue**: SQLite connection or query fails
- **Action**: Skip image, continue with next
- **Recovery**: Automatic (no manual intervention)

### 4. **DynamoDB Errors**
- **Issue**: Result storage fails
- **Action**: Log error, continue processing
- **Recovery**: Automatic (no manual intervention)

## Performance Characteristics

### **Processing Time per Image**
- **Model startup**: ~30-60 seconds (one-time)
- **Meter detection**: ~2-5 seconds
- **OCR processing**: ~1-3 seconds
- **Database operations**: ~0.1-0.5 seconds
- **Total per image**: ~3-8 seconds

### **Batch Processing**
- **Sequential processing**: One image at a time
- **No parallelization**: Ensures model stability
- **Memory usage**: Moderate (image processing in memory)

## Configuration Points

### **AWS Configuration**
- **S3 Bucket**: `smart-meter-ai-demo`
- **Model ARN**: Custom trained Rekognition model
- **DynamoDB Table**: `MeterValidationResults`
- **Region**: us-east-1 (implicit)

### **Processing Configuration**
- **Confidence Threshold**: 95%
- **Tolerance**: 1% or 0.1 units (whichever is larger)
- **Supported Formats**: PNG, JPG, JPEG

### **Database Configuration**
- **SQLite DB**: `manual_readings.db`
- **Table**: `manual_readings`
- **Schema**: (id, image_id, reading, created_at)

## Monitoring and Logging

### **Console Output**
- Model status updates
- Processing progress per image
- Validation results
- Error messages

### **AWS CloudWatch**
- Lambda function metrics (if using Lambda)
- DynamoDB read/write capacity
- S3 access logs

### **Database Monitoring**
- SQLite file size
- Query performance
- Data integrity checks 