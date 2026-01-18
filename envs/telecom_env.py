# envs/telecom_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig
from utils.topology import NetworkGraph # Import module graph vừa tạo

class TelecomEnv(gym.Env):
    def __init__(self, cfg: DictConfig):
        super(TelecomEnv, self).__init__()
        self.cfg = cfg
        
        self.n_cells = cfg.network.num_cells
        self.n_sectors = cfg.network.sectors_per_cell # = 3
        self.total_sectors = self.n_cells * self.n_sectors
        
        # Khởi tạo đồ thị mạng
        self.graph = NetworkGraph(self.n_cells, cfg.network.inter_site_distance)
        
        # --- OBSERVATION SPACE ---
        # Mỗi sector có 3 chỉ số: [Users, Data_Used, Traffic_Load]
        # State shape: (Total_Sectors, 3) + Status của sector (Total_Sectors)
        # Flatten lại cho DRL dễ học: Size = Total_Sectors * 4
        self.observation_space = spaces.Box(
            low=0, high=1000, 
            shape=(self.total_sectors * 4,), 
            dtype=np.float32
        )
        
        # --- ACTION SPACE ---
        # MultiBinary cho từng Sector: 1=On, 0=Off/Sleep
        self.action_space = spaces.MultiBinary(self.total_sectors)
        
        self.reward_function_code = None
        self.last_actions = np.ones(self.total_sectors)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Sinh dữ liệu ngẫu nhiên cho từng Sector
        # Users: số lượng người dùng kết nối
        self.users = np.random.randint(
            self.cfg.traffic.min_users, 
            self.cfg.traffic.max_users, 
            self.total_sectors
        )
        
        # Data Used: Lưu lượng data trung bình mỗi user (Mbps)
        data_demand = np.random.uniform(
            self.cfg.traffic.data_per_user_min,
            self.cfg.traffic.data_per_user_max,
            self.total_sectors
        )
        
        # Traffic Load = Users * Data_Demand
        self.traffic_load = self.users * data_demand
        
        self.sector_status = np.ones(self.total_sectors) # Mặc định bật hết
        return self._get_obs(), {}

    def _get_obs(self):
        # Ghép tất cả thông tin lại thành 1 vector
        # [User_0, Data_0, Load_0, Status_0, User_1, ...]
        obs_matrix = np.stack([
            self.users, 
            self.traffic_load / self.users, # Data used avg
            self.traffic_load,
            self.sector_status
        ], axis=1)
        return obs_matrix.flatten().astype(np.float32)

    def step(self, action):
        cfg = self.cfg
        
        # 1. Xử lý Traffic & Offloading (Dựa trên Graph)
        total_demand = np.sum(self.traffic_load)
        served_traffic = 0
        
        # Loop qua từng Cell
        for cell_id in range(self.n_cells):
            # Lấy indices của 3 sector thuộc cell này
            sec_start = cell_id * self.n_sectors
            sec_end = sec_start + self.n_sectors
            
            cell_sectors_action = action[sec_start:sec_end]
            cell_sectors_load = self.traffic_load[sec_start:sec_end]
            
            # Logic 1: Sector nào bật thì phục vụ traffic của nó
            served_local = np.sum(np.minimum(
                cell_sectors_load * cell_sectors_action, 
                cfg.network.capacity_sector
            ))
            served_traffic += served_local
            
            # Logic 2: Sector nào tắt -> Traffic tràn đi đâu?
            # Ưu tiên 1: Tràn sang sector khác trong cùng Cell (Intra-site)
            # Ưu tiên 2: Tràn sang Cell hàng xóm (Inter-site)
            
            traffic_need_offload = np.sum(cell_sectors_load * (1 - cell_sectors_action))
            
            if traffic_need_offload > 0:
                # Kiểm tra xem cell này còn sector nào bật không?
                active_sectors_in_cell = np.sum(cell_sectors_action)
                
                if active_sectors_in_cell > 0:
                    # Offload nội bộ (đơn giản hóa: coi như sector còn lại gánh được một phần)
                    remaining_cap = (active_sectors_in_cell * cfg.network.capacity_sector) - served_local
                    offloaded = min(traffic_need_offload, max(0, remaining_cap))
                    served_traffic += offloaded
                else:
                    # Cell tắt hoàn toàn -> Offload sang hàng xóm gần nhất
                    # Tìm hàng xóm đang bật (dùng Topology Graph)
                    neighbor_id = self.graph.get_nearest_neighbor(cell_id, exclude_list=[])
                    
                    if neighbor_id is not None:
                        # Giả định hàng xóm gánh được (đơn giản hóa)
                        served_traffic += min(traffic_need_offload, cfg.network.capacity_sector) 
                        # Trong thực tế phải check capacity hàng xóm

        drop_rate = 1.0 - (served_traffic / total_demand) if total_demand > 0 else 0.0
        
        # 2. Tính Năng lượng (Energy Model chi tiết)
        total_power = 0
        for cell_id in range(self.n_cells):
            sec_start = cell_id * self.n_sectors
            active_sectors = np.sum(action[sec_start : sec_start+3])
            
            if active_sectors > 0:
                # Cell Active: Base Power + Power per Sector
                cell_power = cfg.energy.p_base + (active_sectors * cfg.energy.p_sector_active)
            else:
                # Cell Sleep Deep
                cell_power = cfg.energy.p_sleep
            
            total_power += cell_power
            
        # Cộng chi phí chuyển trạng thái
        switches = np.sum(np.abs(action - self.last_actions))
        total_power += switches * cfg.energy.p_switch
        
        # 3. Reward Execution (LLM)
        loc = {
            "power": total_power,
            "drop_rate": drop_rate,
            "switches": switches,
            "users_active": np.sum(self.users),
            "reward": 0.0
        }
        
        if self.reward_function_code:
            try:
                exec(self.reward_function_code, {}, loc)
                reward = loc['reward']
            except:
                reward = -total_power - 1000*drop_rate
        else:
            reward = -total_power - 1000*drop_rate

        # 4. Update Next Step
        self.last_actions = action
        self.sector_status = action
        
        # Traffic biến đổi ngẫu nhiên
        self.users = np.random.randint(cfg.traffic.min_users, cfg.traffic.max_users, self.total_sectors)
        data_demand = np.random.uniform(0.5, 5.0, self.total_sectors)
        self.traffic_load = self.users * data_demand
        
        self.current_step += 1
        terminated = self.current_step >= cfg.rl.max_episode_steps
        
        info = {
            "power": total_power,
            "drop_rate": drop_rate,
            "switches": switches
        }
        
        return self._get_obs(), reward, terminated, False, info