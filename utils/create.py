import sys
import os

# Fix lỗi import đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import hydra
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from omegaconf import DictConfig
from utils.topology import NetworkGraph

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def create_dataset(cfg: DictConfig):
    print(f"--- BẮT ĐẦU SINH DỮ LIỆU KPI VIỄN THÔNG (24h - 15p/bước) ---")
    
    # 1. Cấu hình
    steps = 96  # 24h * 4
    interval_minutes = 15
    start_time_str = "00:00"
    
    # Capacity Sector (để tính PRB Used %)
    # Giả sử capacity_sector trong config là Mbps
    # Đổi sang MB trong 15 phút để ước lượng PRB: 
    # Max MB = (Capacity_Mbps * 900s) / 8
    max_mb_per_15m = (cfg.network.capacity_sector * 60 * 15) / 8
    
    n_cells = cfg.network.num_cells
    sectors_per_cell = cfg.network.sectors_per_cell
    total_sectors = n_cells * sectors_per_cell
    
    topology = NetworkGraph(n_cells, cfg.network.inter_site_distance)
    
    # 2. Sinh mẫu hình sin (Time-series Pattern)
    time_steps_rad = np.linspace(0, 2 * np.pi, steps)
    base_pattern = np.sin(time_steps_rad - np.pi/2) + 1.2 # Đẩy đáy lên > 0.2
    
    # Các list để lưu trữ cho file .pkl (Ma trận)
    traffic_matrix = [] # ps_traffic_mb
    user_matrix = []    # avg_rrc_connected_user
    prb_matrix = []     # prb_dl_used
    
    # List lưu CSV rows
    csv_rows = []
    
    current_time = datetime.strptime(start_time_str, "%H:%M")
    
    print(f"--> Đang tính toán KPIs cho {total_sectors} sectors...")
    
    for t in range(steps):
        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Tạo nhiễu ngẫu nhiên
        noise = np.random.uniform(0.8, 1.2, total_sectors)
        
        # --- 1. Tính User (avg_rrc_connected_user) ---
        current_users = (base_pattern[t] * (cfg.traffic.max_users * 0.8) * noise).astype(int)
        current_users = np.clip(current_users, cfg.traffic.min_users, None)
        
        # --- 2. Tính Traffic (ps_traffic_mb) ---
        # Giả sử mỗi user dùng ngẫu nhiên data (MB/15p)
        # Data demand logic: Giờ cao điểm user dùng nhiều data hơn
        demand_factor = base_pattern[t] # Cao điểm demand cao hơn
        data_per_user = np.random.uniform(2.0, 10.0, total_sectors) * demand_factor
        
        current_traffic_mb = current_users * data_per_user
        
        # --- 3. Tính PRB Used (prb_dl_used) ---
        # Công thức: (Traffic thực tế / Max Capacity) * 100
        # Thêm chút ngẫu nhiên vì PRB phụ thuộc vào nhiễu sóng, khoảng cách user...
        prb_util = (current_traffic_mb / max_mb_per_15m) * 100
        prb_util = prb_util * np.random.uniform(0.9, 1.1, total_sectors)
        prb_util = np.clip(prb_util, 0, 100) # Max 100%
        
        # Lưu vào Matrix (để train AI)
        user_matrix.append(current_users)
        traffic_matrix.append(current_traffic_mb)
        prb_matrix.append(prb_util)
        
        # Lưu vào CSV Row
        for c in range(n_cells):
            # Giả lập tên trạm (eNodeB ID)
            enodeb_id = 10000 + c 
            
            for s in range(sectors_per_cell):
                global_idx = c * sectors_per_cell + s
                
                # Tạo tên Cell Name (Ví dụ: 10000_1, 10000_2)
                cell_name = f"{enodeb_id}_{s+1}"
                
                row = {
                    "timestamp": timestamp_str,
                    "enodeb": enodeb_id,
                    "cell_name": cell_name,
                    "ps_traffic_mb": round(current_traffic_mb[global_idx], 2),
                    "avg_rrc_connected_user": int(current_users[global_idx]),
                    "prb_dl_used": round(prb_util[global_idx], 2)
                }
                csv_rows.append(row)
        
        current_time += timedelta(minutes=interval_minutes)
        
    # Convert sang numpy array
    traffic_matrix = np.array(traffic_matrix)
    user_matrix = np.array(user_matrix)
    prb_matrix = np.array(prb_matrix)
    
    # 3. Lưu file
    folder_name = f"data_Telecom_KPIs_{n_cells}Cells"
    save_path = os.path.join(project_root, "datasets", folder_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Save PKL (Update thêm prb_matrix vào data pack)
    data_pack = {
        "topology": topology,
        "traffic": traffic_matrix, # ps_traffic_mb
        "users": user_matrix,      # avg_rrc
        "prb": prb_matrix,         # prb_used
        "config": cfg
    }
    with open(os.path.join(save_path, "env_data.pkl"), "wb") as f:
        pickle.dump(data_pack, f)
        
    # Save CSV
    csv_path = os.path.join(save_path, "kpi_data.csv")
    df = pd.DataFrame(csv_rows)
    # Sắp xếp lại cột cho đúng thứ tự yêu cầu
    cols = ["timestamp", "enodeb", "cell_name", "ps_traffic_mb", "avg_rrc_connected_user", "prb_dl_used"]
    df = df[cols] 
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Đã tạo dữ liệu KPI chuẩn Viễn thông!")
    print(f"   - File CSV: {csv_path}")
    print(f"   - Mẫu dữ liệu:")
    print(df.head(3))

if __name__ == "__main__":
    create_dataset()
