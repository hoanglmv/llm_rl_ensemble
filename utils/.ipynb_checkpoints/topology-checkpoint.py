import numpy as np
from scipy.spatial import distance_matrix

class NetworkGraph:
    def __init__(self, num_cells, isd=1.5):
        """
        isd (Inter-Site Distance): Khoảng cách giữa 2 trạm (km)
        """
        self.num_cells = num_cells
        self.isd = isd 
        
        # 1. Tạo tọa độ theo dạng Lưới Tổ Ong (Hexagonal Grid)
        self.positions = self._generate_hexagonal_grid(num_cells, isd)
        
        # 2. Tính ma trận khoảng cách (Distance Matrix)
        # Kết quả là ma trận NxN: dist_matrix[i][j] là khoảng cách giữa cell i và j
        self.dist_matrix = distance_matrix(self.positions, self.positions)
        
        # In ra để kiểm tra
        print(f"--- Network Topology Initialized ({num_cells} Cells) ---")
        # print(self.positions)

    def _generate_hexagonal_grid(self, n_points, isd):
        """Sinh tọa độ (x, y) theo hình tổ ong"""
        coords = []
        
        # Tính bán kính lục giác từ ISD (ISD = sqrt(3) * R)
        radius = isd / np.sqrt(3)
        
        # Tạo lưới mở rộng dần từ tâm (0,0)
        # Ring 0: 1 cell (0,0)
        # Ring 1: 6 cells xung quanh
        # Ring 2: 12 cells...
        
        coords.append([0, 0]) # Center cell
        
        ring = 1
        while len(coords) < n_points:
            # Thuật toán sinh lục giác (Hexagon Corner finding)
            for i in range(6): # 6 góc
                # Góc bắt đầu của ring (theo radian)
                angle = np.deg2rad(60 * i + 30) 
                
                # Điểm góc của Ring
                cx = ring * isd * np.cos(angle) # Xấp xỉ vị trí tâm cell
                cy = ring * isd * np.sin(angle)
                
                # Thêm các điểm trên cạnh lục giác (nếu ring > 1)
                # Để đơn giản hóa cho simulation nhỏ (<20 cells), 
                # ta chỉ lấy các đỉnh lục giác của các Ring
                coords.append([cx, cy])
                if len(coords) >= n_points:
                    break
            ring += 1
            
        return np.array(coords[:n_points])

    def get_nearest_neighbor(self, cell_id, exclude_list=[]):
        """Tìm trạm hàng xóm gần nhất (dựa trên dist_matrix đã tính)"""
        distances = self.dist_matrix[cell_id].copy()
        
        # Tự loại bỏ chính mình (khoảng cách = 0)
        distances[cell_id] = np.inf
        
        # Loại bỏ các trạm trong danh sách loại trừ (ví dụ các trạm đã tắt)
        for ex in exclude_list:
            if ex < len(distances):
                distances[ex] = np.inf
        
        # Tìm index có khoảng cách nhỏ nhất
        nearest_id = np.argmin(distances)
        min_dist = distances[nearest_id]
        
        if min_dist == np.inf:
            return None 
            
        return nearest_id

    def get_distance(self, cell_i, cell_j):
        """Lấy khoảng cách cụ thể giữa 2 trạm"""
        return self.dist_matrix[cell_i][cell_j]