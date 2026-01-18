# utils/read.py
import pickle
import os

def load_dataset(dataset_name):
    # Đường dẫn tương đối từ thư mục chạy
    base_path = "datasets"
    file_path = os.path.join(base_path, dataset_name, "env_data.pkl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy dataset tại: {file_path}")
        
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        
    print(f"📂 Đã load dataset: {dataset_name}")
    return data
