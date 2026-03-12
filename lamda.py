import json
import boto3
import pickle
import os
import uuid
import logging
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Inisialisasi client AWS
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
# Menggunakan tabel UserRecommendations sesuai gambar
table = dynamodb.Table('UserRecommendations')

# Konfigurasi (sesuaikan jika perlu)
BUCKET = 'technomart-s3-test'
MODEL_KEY = 'models/hybrid_model.pkl'
LOCAL_MODEL_PATH = '/tmp/hybrid_model.pkl'

# Global cache untuk model
_model_data = None

def load_model():
    global _model_data
    if _model_data is None:
        if not os.path.exists(LOCAL_MODEL_PATH):
            logger.info("Downloading model from S3...")
            s3_client.download_file(BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
        with open(LOCAL_MODEL_PATH, 'rb') as f:
            _model_data = pickle.load(f)
        logger.info("Model loaded.")
    return _model_data

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Parse input dari event
        if 'body' in event and event['body'] is not None:
            if isinstance(event['body'], str):
                try:
                    body = json.loads(event['body'])
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in body: {event['body']}")
                    return {
                        'statusCode': 400,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({'error': 'Invalid JSON in request body'})
                    }
            else:
                body = event['body']
        else:
            body = event

        logger.info(f"Parsed body: {json.dumps(body)}")

        # Ambil user_id dan n
        user_id = body.get('user_id')
        n = body.get('n', 5)

        if user_id is None:
            logger.warning("Missing user_id in request")
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Missing user_id'})
            }

        # Load model (cached)
        model_dict = load_model()
        model = model_dict['model']
        product_map = model_dict['product_map']
        user_to_idx = model_dict['user_to_idx']
        interaction_matrix = model_dict['interaction_matrix']

        # Cek apakah user ada di mapping
        if user_id not in user_to_idx:
            recommendations = []
            logger.info(f"User {user_id} not found, returning empty list")
        else:
            user_idx = user_to_idx[user_id]
            user_row = interaction_matrix[user_idx]
            item_ids, _ = model.recommend(user_idx, user_row, N=n)
            recommendations = [product_map[i] for i in item_ids]
            logger.info(f"Recommendations for user {user_id}: {recommendations}")

        # Response
        response_body = {
            'user_id': user_id,
            'recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        # Simpan ke DynamoDB (tabel UserRecommendations)
        # Partition key: user_id (String)
        item = {
            'user_id': str(user_id),
            'recommendations': recommendations,
            'timestamp': response_body['timestamp']
        }
        # Opsional: tambahkan expiry_time jika TTL diaktifkan di tabel
        # item['expiry_time'] = int((datetime.utcnow() + timedelta(hours=24)).timestamp())
        
        table.put_item(Item=item)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }