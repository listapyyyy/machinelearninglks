# test_endpoint.py
import boto3
import json

# Ganti dengan nama endpoint Anda
endpoint_name = 'recommender-endpoint'

runtime = boto3.client('sagemaker-runtime')

# Contoh user_id (sesuaikan dengan data Anda)
payload = json.dumps({
    'user_id': 'U123',   # pastikan user_id ini ada di data Anda
    'n': 5
})

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=payload
)

result = json.loads(response['Body'].read().decode())
print("Hasil rekomendasi:")
print(result)