# inference.py
import json
import pickle
import numpy as np
from scipy.sparse import load_npz
import os

def model_fn(model_dir):
    """
    Load model, matrix, user_map, product_map dari model_dir.
    Model di-load dari file pickle, matrix dari file npz.
    """
    # Load model dan mapping
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Load matrix interaksi
    matrix_path = os.path.join(model_dir, 'interaction_matrix.npz')
    interaction_matrix = load_npz(matrix_path)  # CSR matrix
    
    # Gabungkan dalam satu dictionary
    data['interaction_matrix'] = interaction_matrix
    return data

def input_fn(request_body, request_content_type):
    """Parse request JSON, ambil user_id dan optional n."""
    if request_content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    input_data = json.loads(request_body)
    user_id = input_data.get('user_id')
    if user_id is None:
        raise ValueError("Missing 'user_id' in request")
    
    n = input_data.get('n', 5)  # default 5 rekomendasi
    return {'user_id': user_id, 'n': n}

def predict_fn(input_dict, model_dict):
    """
    Generate rekomendasi untuk user_id.
    model_dict berisi: model, user_map, product_map, interaction_matrix.
    """
    user_id = input_dict['user_id']
    n = input_dict['n']
    
    model = model_dict['model']
    user_map = model_dict['user_map']
    product_map = model_dict['product_map']
    interaction_matrix = model_dict['interaction_matrix']
    
    # Cari index user
    if user_id not in user_map:
        # Jika user tidak dikenal, kembalikan list kosong
        # Bisa juga diganti dengan rekomendasi populer (lihat catatan)
        return []
    
    user_idx = user_map[user_id]
    
    # Ambil baris matrix untuk user tersebut
    user_row = interaction_matrix[user_idx]
    
    # Panggil model.recommend
    # Catatan: model.recommend mengembalikan (item_ids, scores)
    item_ids, scores = model.recommend(user_idx, user_row, N=n)
    
    # Konversi indeks ke product_id
    recommended_products = [product_map[i] for i in item_ids]
    return recommended_products

def output_fn(prediction, response_content_type):
    """Format output sebagai JSON."""
    return json.dumps({'recommended_products': prediction})