# Smart Meter Reading Validation System

An AI-powered system that automatically extracts meter readings from images using AWS Rekognition and validates them against manual readings stored in a SQLite database.

## 🚀 Features

- **AI-Powered Meter Detection**: Uses custom-trained AWS Rekognition model
- **OCR Text Extraction**: Automatically reads numeric values from meter displays
- **Leading Zero Preservation**: Maintains data integrity for meter readings
- **Batch Processing**: Processes multiple images sequentially
- **Validation System**: Compares AI-extracted readings with manual readings
- **Result Storage**: Stores validation results in DynamoDB
- **Modular Architecture**: Clean, maintainable code structure

## 📁 Project Structure

```
Rekognition/
├── main.py                 # Entry point for batch processing
├── config.py              # Configuration constants
├── aws_clients.py         # AWS service clients
├── db_utils.py            # SQLite database utilities
├── meter_processing.py    # Core meter processing logic
├── requirements.txt       # Python dependencies
├── ref_meter_data.csv     # Reference manual readings
├── manual_readings.db     # SQLite database (auto-generated)
├── EXECUTION_FLOW.md      # Detailed execution flow
└── README.md             # This file
```

## 🛠️ Prerequisites

- Python 3.8+
- AWS Account with appropriate permissions
- AWS CLI configured with credentials
- Access to AWS Rekognition Custom Labels
- Access to S3, DynamoDB services

## 📦 Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**:
   ```bash
   aws configure
   ```

4. **Set up the database** (if not already done):
   ```bash
   # Create a simple database setup script
   python -c "
   import sqlite3
   import pandas as pd
   
   # Create database and table
   conn = sqlite3.connect('manual_readings.db')
   cursor = conn.cursor()
   cursor.execute('''
       CREATE TABLE IF NOT EXISTS manual_readings (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           image_id TEXT NOT NULL UNIQUE,
           reading TEXT NOT NULL,
           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
       )
   ''')
   
   # Load CSV data
   df = pd.read_csv('ref_meter_data.csv', dtype={'meter_reading': str})
   for index, row in df.iterrows():
       cursor.execute('INSERT OR REPLACE INTO manual_readings (image_id, reading) VALUES (?, ?)',
                     (row['imageid'], row['meter_reading']))
   
   conn.commit()
   conn.close()
   print('Database setup complete!')
   "
   ```

## ⚙️ Configuration

Edit `config.py` to match your AWS setup:

```python
# AWS and DB configuration
S3_BUCKET = 'your-s3-bucket-name'           # Your S3 bucket for images
MODEL_ARN = 'your-model-arn'                # Your Rekognition model ARN
DYNAMODB_TABLE = 'your-dynamodb-table'      # Your DynamoDB table name
MIN_CONFIDENCE = 95                         # AI confidence threshold
SQLITE_DB = 'manual_readings.db'            # Local SQLite database
```

## 🚀 Usage

### Basic Usage

1. **Upload images to S3 bucket**:
   ```bash
   aws s3 cp meter1.jpg s3://your-bucket-name/
   aws s3 cp meter2.jpg s3://your-bucket-name/
   ```

2. **Run batch processing**:
   ```bash
   python main.py
   ```

### Expected Output

```
Model already RUNNING
Found 2 images to process

Processing meter1.jpg...
Extracted: 039672 | Manual: 039672 | Validation: PASS

Processing meter2.jpg...
Extracted: 014673 | Manual: 014673 | Validation: PASS
```

## 📊 Data Format

### Input CSV Format (`ref_meter_data.csv`)
```csv
imageid,meter_reading
img01,039672
img02,014673
img03,021043
```

### DynamoDB Result Format
```json
{
  "ImageID": "meter1.jpg",
  "ExtractedReading": 39672.0,
  "Confidence": 98.5,
  "ManualReading": "039672",
  "IsValid": true,
  "Timestamp": 1640995200
}
```

## 🔧 Module Details

### `main.py`
- **Purpose**: Entry point and batch orchestration
- **Functions**: `process_batch_images()`
- **Dependencies**: All other modules

### `config.py`
- **Purpose**: Centralized configuration
- **Contains**: AWS settings, database paths, thresholds

### `aws_clients.py`
- **Purpose**: AWS service initialization
- **Services**: S3, Rekognition, DynamoDB
- **Dependencies**: `config.py`

### `db_utils.py`
- **Purpose**: SQLite database operations
- **Functions**: `get_manual_reading()`
- **Dependencies**: `config.py`

### `meter_processing.py`
- **Purpose**: Core AI processing logic
- **Functions**: 
  - `start_model()` - Model management
  - `extract_meter_data()` - AI extraction
  - `validate_readings()` - Validation logic
  - `store_result()` - Result storage
- **Dependencies**: `config.py`, `aws_clients.py`, `db_utils.py`

## 🐛 Troubleshooting

### Common Issues

#### 1. **AWS Region Error**
```
botocore.exceptions.NoRegionError: You must specify a region.
```
**Solution**: Configure AWS region:
```bash
aws configure set region us-east-1
```

#### 2. **Model Not Running**
```
Model start error: An error occurred (ResourceNotFoundException)
```
**Solution**: Verify MODEL_ARN in `config.py` is correct

#### 3. **Database Connection Error**
```
Database error: no such table: manual_readings
```
**Solution**: Run the database setup script (see Installation step 4)

#### 4. **S3 Access Denied**
```
Access Denied: s3://your-bucket
```
**Solution**: Verify S3 bucket name and permissions

#### 5. **DynamoDB Table Not Found**
```
DynamoDB error: Table not found
```
**Solution**: Create DynamoDB table or verify table name

### Debug Mode

Add debug logging by modifying `config.py`:
```python
DEBUG = True
```

## 📈 Performance Optimization

### For High Volume Processing

1. **Use Lambda Functions**: Convert to serverless for automatic scaling
2. **Implement Parallel Processing**: Process multiple images concurrently
3. **Optimize Model Usage**: Keep model running between batches
4. **Use DynamoDB Streams**: For real-time result processing

### Cost Optimization

1. **Monitor Rekognition Usage**: Track API calls and costs
2. **Optimize Image Sizes**: Compress images before upload
3. **Use S3 Lifecycle Policies**: Archive old images
4. **Implement Caching**: Cache frequently accessed data

## 🔒 Security Considerations

1. **IAM Permissions**: Use least privilege principle
2. **Data Encryption**: Enable S3 and DynamoDB encryption
3. **Network Security**: Use VPC endpoints for AWS services
4. **Access Logging**: Enable CloudTrail for audit trails

## 📝 API Reference

### Core Functions

#### `extract_meter_data(image_key)`
Extracts meter reading from S3 image using AI.

**Parameters**:
- `image_key` (str): S3 object key

**Returns**:
- `tuple`: (extracted_reading, confidence_score)

#### `validate_readings(extracted, manual)`
Validates extracted reading against manual reading.

**Parameters**:
- `extracted`: AI-extracted reading
- `manual`: Manual reference reading

**Returns**:
- `bool`: True if validation passes

#### `get_manual_reading(image_id)`
Retrieves manual reading from database.

**Parameters**:
- `image_id` (str): Image identifier

**Returns**:
- `str`: Manual reading (preserves leading zeros)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the execution flow document
3. Check AWS service status
4. Create an issue with detailed error information

## 🔄 Version History

- **v1.0.0**: Initial release with modular architecture
- **v1.1.0**: Added leading zero preservation
- **v1.2.0**: Enhanced error handling and documentation 