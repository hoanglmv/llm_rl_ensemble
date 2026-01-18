# agents/ppo_agent.py
from stable_baselines3 import PPO
from omegaconf import DictConfig

class DRLAgent:
    def __init__(self, env, cfg: DictConfig):
        self.env = env
        self.cfg = cfg
        self.model = None

    def train(self):
        self.model = PPO("MlpPolicy", self.env, verbose=0)
        # Lấy tham số từ Hydra config
        self.model.learn(total_timesteps=self.cfg.rl.train_timesteps)
        return self.model

    def evaluate(self, episodes=5):
        total_power = 0
        total_drop = 0
        total_switch = 0
        
        steps_per_ep = self.cfg.rl.max_episode_steps
        
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, _, done, _, info = self.env.step(action)
                
                total_power += info['power']
                total_drop += info['drop_rate']
                total_switch += info['switches']
        
        total_steps = episodes * steps_per_ep
        return {
            "avg_power": total_power / total_steps,
            "avg_drop_rate": total_drop / total_steps,
            "avg_switches": total_switch / total_steps
        }