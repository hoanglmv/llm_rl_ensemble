# envs/telecom_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig # Để hint type cho code dễ đọc

class CellOnOffEnv(gym.Env):
    def __init__(self, cfg: DictConfig):
        super(CellOnOffEnv, self).__init__()
        self.cfg = cfg # Lưu config lại để dùng toàn cục
        
        # Truy cập tham số thông qua dấu chấm (dot notation)
        n_sbs = cfg.network.num_sbs
        
        # State & Action spaces
        self.observation_space = spaces.Box(
            low=0, high=200, shape=(2 * n_sbs,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(n_sbs)
        
        self.reward_function_code = None
        self.current_step = 0
        self.last_actions = np.ones(n_sbs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        n_sbs = self.cfg.network.num_sbs
        
        # Dùng config cho traffic
        self.traffic = np.random.uniform(
            self.cfg.traffic.min, 
            self.cfg.traffic.max, 
            n_sbs
        )
        self.sbs_status = np.ones(n_sbs)
        self.last_actions = np.ones(n_sbs)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.traffic, self.sbs_status]).astype(np.float32)

    def step(self, action):
        cfg = self.cfg # Alias cho ngắn gọn
        n_sbs = cfg.network.num_sbs
        
        # 1. Traffic Logic
        total_traffic = np.sum(self.traffic)
        served_traffic = 0
        traffic_to_macro = 0
        
        for i in range(n_sbs):
            if action[i] == 1:
                served = min(self.traffic[i], cfg.network.capacity_sbs)
                served_traffic += served
            else:
                traffic_to_macro += self.traffic[i]
        
        served_by_macro = min(traffic_to_macro, cfg.network.capacity_mbs)
        served_traffic += served_by_macro
        
        dropped_traffic = total_traffic - served_traffic
        drop_rate = dropped_traffic / total_traffic if total_traffic > 0 else 0
        
        # 2. Energy Logic
        active_power = np.sum(action * cfg.energy.p_active + (1 - action) * cfg.energy.p_sleep)
        num_switches = np.sum(np.abs(action - self.last_actions))
        switch_power = num_switches * cfg.energy.p_switch
        total_power = active_power + switch_power
        
        # 3. Reward Execution
        loc = {
            "power": total_power,
            "drop_rate": drop_rate,
            "switches": num_switches,
            "reward": 0.0
        }
        
        if self.reward_function_code:
            try:
                exec(self.reward_function_code, {}, loc)
                reward = loc['reward']
            except:
                reward = -1000.0
        else:
            reward = -total_power - 100 * drop_rate

        # 4. Update
        self.last_actions = action
        self.sbs_status = action
        self.traffic = np.random.uniform(cfg.traffic.min, cfg.traffic.max, n_sbs)
        
        self.current_step += 1
        terminated = self.current_step >= cfg.rl.max_episode_steps
        
        info = {
            "power": total_power,
            "drop_rate": drop_rate,
            "switches": num_switches
        }
        
        return self._get_obs(), reward, terminated, False, info