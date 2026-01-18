import sys
import os

# [FIX LỖI IMPORT] Thêm thư mục gốc dự án vào đường dẫn tìm kiếm của Python
# Giúp file này tìm thấy module 'utils' ngay cả khi chạy trực tiếp
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import hydra
import pickle
import numpy as np
from omegaconf import DictConfig
from utils.topology import NetworkGraph # Giờ dòng này sẽ chạy ngon lành

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def create_dataset(cfg: DictConfig):
    print(f"--- Bắt đầu sinh dữ liệu ---")
    
    # 1. Tạo Topology
    n_cells = cfg.network.num_cells
    topology = NetworkGraph(n_cells, cfg.network.inter_site_distance)
    
    # 2. Sinh Traffic Time-series
    steps = cfg.traffic.simulation_steps
    total_sectors = n_cells * cfg.network.sectors_per_cell
    
    time_steps = np.linspace(0, 2 * np.pi, steps)
    base_traffic = np.sin(time_steps - np.pi/2) + 1.2 
    
    traffic_data = []
    user_data = []
    
    for t in range(steps):
        noise = np.random.uniform(0.8, 1.2, total_sectors)
        
        current_users = (base_traffic[t] * cfg.traffic.max_users * noise).astype(int)
        current_users = np.clip(current_users, cfg.traffic.min_users, None)
        
        data_demand = np.random.uniform(
            cfg.traffic.data_per_user_min, 
            cfg.traffic.data_per_user_max, 
            total_sectors
        )
        
        load = current_users * data_demand
        
        user_data.append(current_users)
        traffic_data.append(load)
        
    traffic_data = np.array(traffic_data) 
    user_data = np.array(user_data)       
    
    # 3. Tạo tên folder đặc trưng
    folder_name = f"data_C{n_cells}_S{steps}_U{cfg.traffic.max_users}"
    # Sửa đường dẫn lưu cho chuẩn
    save_path = os.path.join(project_root, "datasets", folder_name)
    
    os.makedirs(save_path, exist_ok=True)
    
    # 4. Lưu file
    data_pack = {
        "topology": topology,
        "traffic": traffic_data,
        "users": user_data,
        "config": cfg
    }
    
    file_path = os.path.join(save_path, "env_data.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data_pack, f)
        
    print(f"✅ Đã lưu dữ liệu tại: {file_path}")
    print(f"   - Topology: {n_cells} cells")
    print(f"   - Traffic Shape: {traffic_data.shape}")

if __name__ == "__main__":
    create_dataset()