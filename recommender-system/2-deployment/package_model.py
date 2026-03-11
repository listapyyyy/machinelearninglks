import boto3
import tarfile
import os

# Konfigurasi
bucket = 'technomart-s3-test'
model_pkl_key = 'models/recommender_model.pkl'
matrix_npz_key = 'models/interaction_matrix.npz'
output_tar_key = 'models/recommender_model.tar.gz'

# Inisialisasi S3 client
s3 = boto3.client('s3')

print("Downloading model and matrix from S3...")
s3.download_file(bucket, model_pkl_key, 'model.pkl')
s3.download_file(bucket, matrix_npz_key, 'interaction_matrix.npz')

print("Creating tarball...")
with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('model.pkl')
    tar.add('interaction_matrix.npz')

print(f"Uploading tarball to s3://{bucket}/{output_tar_key}")
s3.upload_file('model.tar.gz', bucket, output_tar_key)

print("Done. Tarball ready for deployment.")