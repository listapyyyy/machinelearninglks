# app.py
import pickle
from flask import Flask, request, jsonify
from scipy.sparse import load_npz

# ==========================================
# LOAD MODEL DAN DATA (dieksekusi sekali saat server start)
# ==========================================
print("Memuat model dan matrix...")
try:
    with open('recommender_model.pkl', 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    user_map = data['user_map']          # index -> user_id
    product_map = data['product_map']    # index -> product_id
    matrix = load_npz('interaction_matrix.npz')
    user_to_idx = {v: k for k, v in user_map.items()}  # mapping user_id -> index
    print("✅ Model dan matrix berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    exit(1)

# ==========================================
# INISIALISASI FLASK
# ==========================================
app = Flask(__name__)

# ==========================================
# ENDPOINT REKOMENDASI (POST)
# ==========================================
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Menerima JSON: {"user_id": <user_id>, "n": <jumlah_rekomendasi>}
    Mengembalikan daftar product_id rekomendasi.
    """
    try:
        content = request.get_json()
        if not content:
            return jsonify({'error': 'Request body harus JSON'}), 400

        user_id = content.get('user_id')
        n = content.get('n', 5)

        if user_id is None:
            return jsonify({'error': 'Field "user_id" wajib diisi'}), 400

        # Konversi tipe jika perlu (user_id bisa int atau string)
        # Sesuaikan dengan tipe asli di dataset Anda
        # Di sini kita asumsikan user_id bisa berupa int atau string
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)

        if user_id not in user_to_idx:
            return jsonify({'recommended_products': []})  # user tidak dikenal

        user_idx = user_to_idx[user_id]
        user_row = matrix[user_idx]
        item_ids, _ = model.recommend(user_idx, user_row, N=n)
        recommended_products = [product_map[i] for i in item_ids]

        return jsonify({'recommended_products': recommended_products})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# ENDPOINT LIHAT SEMUA USER (GET)
# ==========================================
@app.route('/users', methods=['GET'])
def get_users():
    """Mengembalikan daftar semua user_id."""
    all_users = list(user_to_idx.keys())
    return jsonify({'total': len(all_users), 'users': all_users})

# ==========================================
# ENDPOINT LIHAT SEMUA PRODUK (GET)
# ==========================================
@app.route('/products', methods=['GET'])
def get_products():
    """Mengembalikan daftar semua product_id."""
    all_products = list(product_map.values())
    return jsonify({'total': len(all_products), 'products': all_products})

# ==========================================
# ENDPOINT LIHAT INTERAKSI USER (GET)
# ==========================================
@app.route('/interactions/<user_id>', methods=['GET'])
def get_user_interactions(user_id):
    """
    Mengembalikan daftar produk yang pernah diinteraksi oleh user tertentu.
    """
    # Konversi user_id ke tipe yang sesuai
    try:
        if user_id.isdigit():
            user_id = int(user_id)
    except:
        pass

    if user_id not in user_to_idx:
        return jsonify({'error': 'User not found'}), 404

    user_idx = user_to_idx[user_id]
    user_row = matrix[user_idx]
    product_indices = user_row.indices
    product_ids = [product_map[i] for i in product_indices]

    return jsonify({
        'user_id': user_id,
        'total_interactions': len(product_ids),
        'products': product_ids
    })

# ==========================================
# ENDPOINT STATISTIK (GET)
# ==========================================
@app.route('/stats', methods=['GET'])
def get_stats():
    """Statistik dasar dataset."""
    total_users = len(user_map)
    total_products = len(product_map)
    total_interactions = matrix.nnz
    density = (total_interactions / (total_users * total_products)) * 100
    return jsonify({
        'total_users': total_users,
        'total_products': total_products,
        'matrix_shape': [matrix.shape[0], matrix.shape[1]],
        'total_interactions': total_interactions,
        'density': f"{density:.4f}%"
    })

# ==========================================
# JALANKAN SERVER
# ==========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
