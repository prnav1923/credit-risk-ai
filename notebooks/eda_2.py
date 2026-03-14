import pandas as pd
import io
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
obj = s3.get_object(Bucket=os.getenv('S3_BUCKET_NAME'), Key='processed/train.csv')
df = pd.read_csv(io.BytesIO(obj['Body'].read()), nrows=5)
print(df.columns.tolist())