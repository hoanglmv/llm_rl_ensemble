# utils/create.py
import hydra
import os
import pickle
import numpy as np
from omegaconf import DictConfig
from utils.topology import NetworkGraph

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def create_dataset(cfg: DictConfig):
    print(f"--- Bắt đầu sinh dữ liệu ---")
    
    # 1. Tạo Topology
    n_cells = cfg.network.num_cells
    topology = NetworkGraph(n_cells, cfg.network.inter_site_distance)
    
    # 2. Sinh Traffic Time-series (Shape: [Steps, Total_Sectors])
    # Giả lập traffic hình Sin (ngày cao, đêm thấp)
    steps = cfg.traffic.simulation_steps
    total_sectors = n_cells * cfg.network.sectors_per_cell
    
    time_steps = np.linspace(0, 2 * np.pi, steps)
    base_traffic = np.sin(time_steps - np.pi/2) + 1.2 # Đẩy lên > 0
    
    # Thêm nhiễu ngẫu nhiên cho từng sector
    traffic_data = []
    user_data = []
    
    for t in range(steps):
        # Hệ số ngẫu nhiên cho từng sector tại thời điểm t
        noise = np.random.uniform(0.8, 1.2, total_sectors)
        
        # Số user biến đổi theo hình sin
        current_users = (base_traffic[t] * cfg.traffic.max_users * noise).astype(int)
        current_users = np.clip(current_users, cfg.traffic.min_users, None)
        
        # Data demand ngẫu nhiên
        data_demand = np.random.uniform(
            cfg.traffic.data_per_user_min, 
            cfg.traffic.data_per_user_max, 
            total_sectors
        )
        
        load = current_users * data_demand
        
        user_data.append(current_users)
        traffic_data.append(load)
        
    traffic_data = np.array(traffic_data) # Shape: (24, 15)
    user_data = np.array(user_data)       # Shape: (24, 15)
    
    # 3. Tạo tên folder đặc trưng
    # VD: dataset_5cells_24steps
    folder_name = f"data_C{n_cells}_S{steps}_U{cfg.traffic.max_users}"
    save_path = os.path.join(hydra.utils.get_original_cwd(), "datasets", folder_name)
    
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
