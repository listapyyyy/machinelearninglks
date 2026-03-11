# ============================================
# CONFIGURATION
# ============================================
BUCKET_NAME = "technomart-s3-test"                 # Ganti dengan nama bucket Anda
USER_INTERACTIONS_PATH = "processed-data/interactions-processed"
TRANSACTION_HISTORY_PATH = "processed-data/transaction-processed"
PRODUCT_CATALOG_PATH = "processed-data/catalog-processed"
MODEL_OUTPUT_PATH = "models/recommender_model.pkl"      # Lokasi model di S3
MATRIX_OUTPUT_PATH = "models/interaction_matrix.npz"    # Lokasi matrix di S3
MODEL_TAR_GZ_PATH = "models/recommender_model.tar.gz"   # Tarball untuk deployment