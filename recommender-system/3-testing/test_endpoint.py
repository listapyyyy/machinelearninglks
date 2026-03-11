import boto3
import json
import argparse

# Konfigurasi
endpoint_name = 'recommender-endpoint'   # ganti jika berbeda

def test_endpoint(user_id, n=5):
    runtime = boto3.client('sagemaker-runtime')
    
    payload = json.dumps({
        'user_id': user_id,
        'n': n
    })
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    
    result = json.loads(response['Body'].read().decode())
    print(f"Recommendations for user {user_id}:")
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_id', type=str, required=True, help='User ID to get recommendations for')
    parser.add_argument('--n', type=int, default=5, help='Number of recommendations')
    args = parser.parse_args()
    
    test_endpoint(args.user_id, args.n)