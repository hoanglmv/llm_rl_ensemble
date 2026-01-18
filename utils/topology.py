# utils/topology.py
import numpy as np
from scipy.spatial import distance_matrix

class NetworkGraph:
    def __init__(self, num_cells, isd=1.5):
        self.num_cells = num_cells
        self.isd = isd # Inter-site distance (km)
        
        # 1. Tạo tọa độ (Giả lập Hexagonal hoặc Random)
        # Ở đây dùng random cho đơn giản, nhưng nhân với ISD để ra khoảng cách thực
        self.positions = np.random.rand(num_cells, 2) * (np.sqrt(num_cells) * isd)
        
        # 2. Tính ma trận khoảng cách giữa các trạm
        # shape: (num_cells, num_cells)
        self.dist_matrix = distance_matrix(self.positions, self.positions)
        
        # Đánh dấu neighbor (ví dụ: các trạm cách nhau < 2.5km là hàng xóm)
        self.adj_matrix = (self.dist_matrix < 2.5 * isd) & (self.dist_matrix > 0)

    def get_nearest_neighbor(self, cell_id, exclude_list=[]):
        """Tìm trạm hàng xóm gần nhất đang BẬT để offload traffic"""
        distances = self.dist_matrix[cell_id].copy()
        
        # Đặt khoảng cách các trạm bị loại trừ (đang tắt) thành vô cực
        for ex in exclude_list:
            distances[ex] = np.inf
        
        # Tìm trạm gần nhất
        nearest_id = np.argmin(distances)
        min_dist = distances[nearest_id]
        
        if min_dist == np.inf:
            return None # Không còn hàng xóm nào
        return nearest_id