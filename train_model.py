import pandas as pd
import numpy as np
import boto3
import pickle
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import config

# ==============================
# LOAD DATA FROM S3
# ==============================
bucket = config.BUCKET_NAME

ui_path = f"s3://{bucket}/{config.USER_INTERACTIONS_PATH}"
th_path = f"s3://{bucket}/{config.TRANSACTION_HISTORY_PATH}"
pc_path = f"s3://{bucket}/{config.PRODUCT_CATALOG_PATH}"

print("Loading datasets from S3...")

ui_df = pd.read_parquet(ui_path)
th_df = pd.read_parquet(th_path)
pc_df = pd.read_parquet(pc_path)

# ==============================
# PREPARE INTERACTION DATA
# ==============================
print("Preparing interaction dataset...")

ui_df = ui_df[['user_id','product_id']]
th_df = th_df[['user_id','product_id']]

interaction_df = pd.concat([ui_df, th_df])
interaction_df["interaction"] = 1

# Encode user dan product
interaction_df["user_idx"] = interaction_df["user_id"].astype("category").cat.codes
interaction_df["product_idx"] = interaction_df["product_id"].astype("category").cat.codes

user_map = dict(enumerate(interaction_df["user_id"].astype("category").cat.categories))
product_map = dict(enumerate(interaction_df["product_id"].astype("category").cat.categories))

# ==============================
# CREATE SPARSE MATRIX
# ==============================
print("Building sparse matrix...")

matrix = coo_matrix(
    (
        interaction_df["interaction"],
        (interaction_df["user_idx"], interaction_df["product_idx"])
    )
)

# ==============================
# TRAIN MODEL
# ==============================
print("Training ALS recommendation model...")

model = AlternatingLeastSquares(
    factors=50,
    regularization=0.1,
    iterations=20
)

model.fit(matrix)

# ==============================
# SAVE MODEL
# ==============================
print("Saving model...")

model_data = {
    "model": model,
    "user_map": user_map,
    "product_map": product_map
}

local_model_file = "recommender_model.pkl"

with open(local_model_file, "wb") as f:
    pickle.dump(model_data, f)

# ==============================
# UPLOAD MODEL TO S3
# ==============================
print("Uploading model to S3...")

s3 = boto3.client("s3")

s3.upload_file(
    local_model_file,
    bucket,
    config.MODEL_OUTPUT_PATH
)

print("Training complete. Model saved to S3.")