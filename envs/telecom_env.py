import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

class TelecomEnv(gym.Env):
    # [QUAN TRỌNG] Thêm tham số data_pack=None vào đây
    def __init__(self, cfg: DictConfig, data_pack=None):
        super(TelecomEnv, self).__init__()
        self.cfg = cfg
        
        # 1. Xử lý Data Pack (Dataset)
        if data_pack:
            # Nếu có dataset, dùng topology và traffic từ file
            self.graph = data_pack['topology']
            self.traffic_matrix = data_pack['traffic'] # Shape: (Steps, Sectors)
            self.user_matrix = data_pack['users']      # Shape: (Steps, Sectors)
            self.max_data_steps = self.traffic_matrix.shape[0]
            print(f"   --> Env đã load {self.max_data_steps} bước dữ liệu từ Dataset.")
        else:
            # Nếu không có dataset, báo lỗi vì hệ thống mới yêu cầu phải có
            raise ValueError("LỖI: Environment yêu cầu phải có 'data_pack' (chạy utils/create.py trước)")

        self.n_cells = cfg.network.num_cells
        self.n_sectors = cfg.network.sectors_per_cell 
        self.total_sectors = self.n_cells * self.n_sectors
        
        # --- Config Spaces ---
        self.observation_space = spaces.Box(
            low=0, high=100000, shape=(self.total_sectors * 4,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(self.total_sectors)
        
        self.reward_function_code = None
        self.last_actions = np.ones(self.total_sectors)
        
        # [FIX] Khởi tạo biến đếm
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Lấy dữ liệu tại bước đầu tiên (t=0)
        self.current_traffic = self.traffic_matrix[0]
        self.current_users = self.user_matrix[0]
        
        self.sector_status = np.ones(self.total_sectors)
        self.last_actions = np.ones(self.total_sectors)
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.stack([
            self.current_users,
            # Tránh chia cho 0
            self.current_traffic / (self.current_users + 1e-9),
            self.current_traffic,
            self.sector_status
        ], axis=1)
        return obs.flatten().astype(np.float32)

    def step(self, action):
        cfg = self.cfg
        
        # 1. Cập nhật dữ liệu Traffic theo thời gian thực (Time-series)
        # Dùng phép chia lấy dư (%) để lặp lại dữ liệu nếu train lâu hơn 24h
        t_idx = self.current_step % self.max_data_steps
        self.current_traffic = self.traffic_matrix[t_idx]
        self.current_users = self.user_matrix[t_idx]
        
        # 2. Tính toán Traffic phục vụ & Drop Rate
        # (Logic đơn giản hóa để chạy nhanh, bạn có thể thay bằng logic topology phức tạp nếu cần)
        
        # Capacity thực tế = Capacity Sector * Trạng thái Bật/Tắt
        available_capacity = action * cfg.network.capacity_sector
        
        # Traffic được phục vụ = Min(Nhu cầu, Khả năng đáp ứng)
        served_traffic = np.minimum(self.current_traffic, available_capacity)
        
        # Tính tổng
        total_demand_step = np.sum(self.current_traffic)
        total_served_step = np.sum(served_traffic)
        
        # Drop Rate = Phần không được phục vụ / Tổng nhu cầu
        if total_demand_step > 0:
            drop_rate = 1.0 - (total_served_step / total_demand_step)
        else:
            drop_rate = 0.0
            
        # 3. Tính toán Năng lượng
        # Công suất nền (cho các Cell có ít nhất 1 sector bật)
        active_cells = 0
        for i in range(self.n_cells):
            # Kiểm tra xem cell thứ i có sector nào bật không
            start = i * self.n_sectors
            end = start + self.n_sectors
            if np.sum(action[start:end]) > 0:
                active_cells += 1
                
        active_sectors = np.sum(action)
        
        # P_Total = (Số Cell bật * P_Base) + (Số Sector bật * P_Sector)
        total_power = (active_cells * cfg.energy.p_base) + (active_sectors * cfg.energy.p_sector_active)
        
        # Cộng phạt chuyển đổi trạng thái
        switches = np.sum(np.abs(action - self.last_actions))
        total_power += switches * cfg.energy.p_switch

        # 4. Tính Reward (LLM Dynamic Reward)
        loc = {
            "power": total_power,
            "drop_rate": drop_rate,
            "switches": switches,
            "users_active": np.sum(self.current_users),
            "reward": 0.0
        }
        
        # Thực thi code do LLM viết
        if self.reward_function_code:
            try:
                exec(self.reward_function_code, {}, loc)
                reward = loc['reward']
            except:
                reward = -total_power - 1000 * drop_rate # Fallback
        else:
            reward = -total_power - 1000 * drop_rate

        # 5. Update trạng thái
        self.last_actions = action
        self.sector_status = action
        self.current_step += 1
        
        # Kết thúc khi hết dữ liệu trong file dataset (hoặc max_step config)
        terminated = self.current_step >= self.max_data_steps
        
        info = {
            "power": total_power, 
            "drop_rate": drop_rate, 
            "switches": switches
        }
        
        return self._get_obs(), reward, terminated, False, info
