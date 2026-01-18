import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig
from utils.topology import NetworkGraph

class TelecomEnv(gym.Env):
    def __init__(self, cfg: DictConfig):
        super(TelecomEnv, self).__init__()
        self.cfg = cfg
        
        self.n_cells = cfg.network.num_cells
        self.n_sectors = cfg.network.sectors_per_cell 
        self.total_sectors = self.n_cells * self.n_sectors
        
        # Khởi tạo đồ thị mạng
        self.graph = NetworkGraph(self.n_cells, cfg.network.inter_site_distance)
        
        # --- OBSERVATION SPACE ---
        self.observation_space = spaces.Box(
            low=0, high=10000, 
            shape=(self.total_sectors * 4,), 
            dtype=np.float32
        )
        
        # --- ACTION SPACE ---
        self.action_space = spaces.MultiBinary(self.total_sectors)
        
        self.reward_function_code = None
        self.last_actions = np.ones(self.total_sectors)
        
        # [QUAN TRỌNG: SỬA LỖI Ở ĐÂY]
        self.current_step = 0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # [QUAN TRỌNG: RESET BIẾN ĐẾM]
        self.current_step = 0
        
        # Sinh dữ liệu ngẫu nhiên
        self.users = np.random.randint(
            self.cfg.traffic.min_users, 
            self.cfg.traffic.max_users, 
            self.total_sectors
        )
        
        data_demand = np.random.uniform(
            self.cfg.traffic.data_per_user_min,
            self.cfg.traffic.data_per_user_max,
            self.total_sectors
        )
        
        self.traffic_load = self.users * data_demand
        self.sector_status = np.ones(self.total_sectors)
        self.last_actions = np.ones(self.total_sectors)
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs_matrix = np.stack([
            self.users, 
            self.traffic_load / (self.users + 1e-9), 
            self.traffic_load,
            self.sector_status
        ], axis=1)
        return obs_matrix.flatten().astype(np.float32)

    def step(self, action):
        cfg = self.cfg
        
        # 1. Logic Traffic
        total_demand = np.sum(self.traffic_load)
        served_traffic = 0
        
        for cell_id in range(self.n_cells):
            sec_start = cell_id * self.n_sectors
            sec_end = sec_start + self.n_sectors
            
            cell_sectors_action = action[sec_start:sec_end]
            cell_sectors_load = self.traffic_load[sec_start:sec_end]
            
            served_local = np.sum(np.minimum(
                cell_sectors_load * cell_sectors_action, 
                cfg.network.capacity_sector
            ))
            served_traffic += served_local
            
            traffic_need_offload = np.sum(cell_sectors_load * (1 - cell_sectors_action))
            
            if traffic_need_offload > 0:
                active_sectors_in_cell = np.sum(cell_sectors_action)
                if active_sectors_in_cell > 0:
                    remaining_cap = (active_sectors_in_cell * cfg.network.capacity_sector) - served_local
                    served_traffic += min(traffic_need_offload, max(0, remaining_cap))
                else:
                    neighbor_id = self.graph.get_nearest_neighbor(cell_id, exclude_list=[])
                    if neighbor_id is not None:
                        served_traffic += min(traffic_need_offload, cfg.network.capacity_sector)

        drop_rate = 1.0 - (served_traffic / total_demand) if total_demand > 0 else 0.0
        
        # 2. Logic Power
        total_power = 0
        for cell_id in range(self.n_cells):
            sec_start = cell_id * self.n_sectors
            active_sectors = np.sum(action[sec_start : sec_start+3])
            
            if active_sectors > 0:
                cell_power = cfg.energy.p_base + (active_sectors * cfg.energy.p_sector_active)
            else:
                cell_power = cfg.energy.p_sleep
            total_power += cell_power
            
        switches = np.sum(np.abs(action - self.last_actions))
        total_power += switches * cfg.energy.p_switch
        
        # 3. Reward LLM
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
                reward = -total_power - 1000 * drop_rate
        else:
            reward = -total_power - 1000 * drop_rate

        # 4. Update Next Step
        self.last_actions = action
        self.sector_status = action
        
        self.users = np.random.randint(cfg.traffic.min_users, cfg.traffic.max_users, self.total_sectors)
        data_demand = np.random.uniform(cfg.traffic.data_per_user_min, cfg.traffic.data_per_user_max, self.total_sectors)
        self.traffic_load = self.users * data_demand
        
        # [SỬA LỖI] Biến này giờ đã được khởi tạo
        self.current_step += 1
        terminated = self.current_step >= cfg.rl.max_episode_steps
        
        info = {
            "power": total_power,
            "drop_rate": drop_rate,
            "switches": switches
        }
        
        return self._get_obs(), reward, terminated, False, info
