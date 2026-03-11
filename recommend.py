import boto3
import pickle
import numpy as np
import config
from scipy.sparse import csr_matrix

bucket = config.BUCKET_NAME
model_key = config.MODEL_OUTPUT_PATH

print("Downloading model from S3...")

s3 = boto3.client("s3")
s3.download_file(bucket, model_key, "model.pkl")

with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
user_map = data["user_map"]
product_map = data["product_map"]

# ==============================
# EXAMPLE RECOMMENDATION
# ==============================

def recommend_for_user(user_index, matrix, n=5):
    ids, scores = model.recommend(user_index, matrix[user_index], N=n)
    products = [product_map[i] for i in ids]
    return products

print("Model loaded. Ready for recommendation.")