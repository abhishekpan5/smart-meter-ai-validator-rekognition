import boto3
from config import S3_BUCKET, DYNAMODB_TABLE

s3 = boto3.resource('s3')
rekognition = boto3.client('rekognition')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE) 