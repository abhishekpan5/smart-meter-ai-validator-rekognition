# Automatic Image Processing Setup

## Overview
This setup will automatically trigger meter reading validation whenever an image is uploaded to your S3 bucket.

## Option 1: AWS Lambda + S3 Event Notifications (Recommended)

### Step 1: Create Lambda Function
1. **Create Lambda Function Package:**
   ```bash
   # Create deployment package
   mkdir lambda-package
   cp lambda_function.py lambda-package/
   cp requirements.txt lambda-package/
   
   # Install dependencies
   cd lambda-package
   pip install -r requirements.txt -t .
   
   # Create ZIP file
   zip -r lambda-deployment.zip .
   ```

2. **Deploy via AWS CLI:**
   ```bash
   # Create Lambda function
   aws lambda create-function \
     --function-name meter-validation-lambda \
     --runtime python3.9 \
     --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
     --handler lambda_function.lambda_handler \
     --zip-file fileb://lambda-deployment.zip \
     --timeout 300 \
     --memory-size 512
   ```

### Step 2: Set Up S3 Event Notifications
1. **Configure S3 Bucket Notifications:**
   ```bash
   # Add Lambda permission for S3
   aws lambda add-permission \
     --function-name meter-validation-lambda \
     --statement-id s3-trigger \
     --action lambda:InvokeFunction \
     --principal s3.amazonaws.com \
     --source-arn arn:aws:s3:::your-image-bucket
   
   # Configure S3 notification
   aws s3api put-bucket-notification-configuration \
     --bucket your-image-bucket \
     --notification-configuration '{
       "LambdaConfigurations": [
         {
           "Id": "meter-validation-trigger",
           "LambdaFunctionArn": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:meter-validation-lambda",
           "Events": ["s3:ObjectCreated:*"],
           "Filter": {
             "Key": {
               "FilterRules": [
                 {"Name": "suffix", "Value": ".jpg"},
                 {"Name": "suffix", "Value": ".jpeg"},
                 {"Name": "suffix", "Value": ".png"}
               ]
             }
           }
         }
       ]
     }'
   ```

### Step 3: Deploy via CloudFormation (Alternative)
```bash
# Deploy the CloudFormation stack
aws cloudformation create-stack \
  --stack-name meter-validation-stack \
  --template-body file://lambda_deployment.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

## Option 2: EventBridge + Step Functions

### Step 1: Create EventBridge Rule
```json
{
  "Name": "meter-image-upload-rule",
  "EventPattern": {
    "source": ["aws.s3"],
    "detail-type": ["Object Created"],
    "detail": {
      "bucket": {
        "name": ["your-image-bucket"]
      },
      "object": {
        "key": [{"suffix": ".jpg"}, {"suffix": ".jpeg"}, {"suffix": ".png"}]
      }
    }
  },
  "Targets": [
    {
      "Id": "meter-validation-target",
      "Arn": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:meter-validation-lambda"
    }
  ]
}
```

## Option 3: S3 + SQS + Lambda (For High Volume)

### Step 1: Create SQS Queue
```bash
# Create SQS queue
aws sqs create-queue \
  --queue-name meter-validation-queue \
  --attributes '{
    "VisibilityTimeout": "300",
    "MessageRetentionPeriod": "1209600"
  }'
```

### Step 2: Configure S3 to SQS
```bash
# Configure S3 notification to SQS
aws s3api put-bucket-notification-configuration \
  --bucket your-image-bucket \
  --notification-configuration '{
    "QueueConfigurations": [
      {
        "Id": "meter-validation-queue-trigger",
        "QueueArn": "arn:aws:sqs:us-east-1:YOUR_ACCOUNT_ID:meter-validation-queue",
        "Events": ["s3:ObjectCreated:*"],
        "Filter": {
          "Key": {
            "FilterRules": [
              {"Name": "suffix", "Value": ".jpg"},
              {"Name": "suffix", "Value": ".jpeg"},
              {"Name": "suffix", "Value": ".png"}
            ]
          }
        }
      }
    ]
  }'
```

## Testing the Setup

### Test Image Upload
```bash
# Upload a test image
aws s3 cp test-meter.jpg s3://your-image-bucket/

# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/meter-validation-lambda"

# Check DynamoDB for results
aws dynamodb scan --table-name MeterValidationResults
```

### Monitor Processing
```bash
# Check Lambda metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=meter-validation-lambda \
  --start-time $(date -d '1 hour ago' --iso-8601=seconds) \
  --end-time $(date --iso-8601=seconds) \
  --period 300 \
  --statistics Sum
```

## Important Considerations

### 1. **SQLite Database Issue**
The current Lambda function tries to use SQLite, which won't work in Lambda. You need to:
- **Option A**: Move manual readings to DynamoDB
- **Option B**: Use RDS/Aurora for manual readings
- **Option C**: Store manual readings in S3 as JSON files

### 2. **Lambda Layer for Dependencies**
For PIL and other heavy dependencies:
```bash
# Create Lambda layer
mkdir -p lambda-layer/python
pip install pillow boto3 -t lambda-layer/python/
cd lambda-layer
zip -r lambda-layer.zip python/
```

### 3. **Cold Start Optimization**
- Keep Lambda function warm with scheduled invocations
- Use provisioned concurrency for consistent performance
- Consider using Lambda Extensions for model caching

### 4. **Error Handling**
- Set up CloudWatch alarms for Lambda errors
- Implement dead letter queues for failed processing
- Add retry logic for transient failures

## Cost Optimization

### 1. **Lambda Configuration**
- Set appropriate timeout (300 seconds max)
- Optimize memory allocation (512MB recommended)
- Use provisioned concurrency for consistent performance

### 2. **Rekognition Costs**
- Monitor model usage and stop when not needed
- Consider batch processing for high volumes
- Use appropriate confidence thresholds

### 3. **S3 Lifecycle Policies**
- Move processed images to cheaper storage classes
- Set up automatic deletion for old images
- Use S3 Intelligent Tiering for cost optimization 