# Technomart Recommendation System

Machine Learning recommendation system menggunakan AWS data pipeline.

Architecture:

AWS Glue
↓
Amazon S3 (processed-data)
↓
SageMaker Training Script
↓
Recommendation Model
↓
Amazon S3 models/

Dataset:

- product_catalog
- user_interactions
- transaction_history
- user_profiles

Algorithm:

Collaborative Filtering (ALS)

Folder Structure:

technomart-recommendation
│
├── train_model.py
├── recommend.py
├── config.py
├── requirements.txt
└── notebooks

Usage:

1 Install library

pip install -r requirements.txt

2 Train model

python train_model.py

3 Model akan tersimpan di

s3://technomart-s3-test/models/recommender_model.pkl