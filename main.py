# main.py
import hydra
from omegaconf import DictConfig, OmegaConf

from envs.telecom_env import CellOnOffEnv
from llm.reward_designer import LLMRewardDesigner
from agents.ppo_agent import DRLAgent

# Chỉ định đường dẫn tới folder chứa config
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # In ra config để kiểm tra (tùy chọn)
    print(OmegaConf.to_yaml(cfg))
    
    # 1. Khởi tạo với Config object
    env = CellOnOffEnv(cfg)     # Truyền cfg vào Env
    llm = LLMRewardDesigner()   
    agent = DRLAgent(env, cfg)  # Truyền cfg vào Agent
    
    feedback = "Khởi động hệ thống."
    
    # Lấy số vòng lặp từ config
    rounds = cfg.llm.simulation_rounds
    
    for i in range(rounds):
        print(f"\n{'='*15} ROUND {i+1}/{rounds} {'='*15}")
        
        # Bước 1: LLM sinh code
        reward_code = llm.generate_code(feedback)
        print(f"--> Reward Function:\n{reward_code}")
        
        # Bước 2: Update Env
        env.reward_function_code = reward_code
        
        # Bước 3: Train
        print("--> Training DRL Agent...")
        agent.train()
        
        # Bước 4: Evaluate
        metrics = agent.evaluate()
        p = metrics['avg_power']
        d = metrics['avg_drop_rate'] * 100
        s = metrics['avg_switches']
        
        print(f"--> Result: Power={p:.1f}W | Drop={d:.2f}% | Switch={s:.2f}")
        
        # Logic Feedback dùng ngưỡng từ config
        threshold = cfg.rl.threshold_drop
        
        if metrics['avg_drop_rate'] > threshold:
            reason = f"Drop Rate {d:.2f}% > {threshold*100}% (Threshold)."
            feedback = f"BAD. {reason}"
        else:
            feedback = "GOOD. Tiếp tục tối ưu năng lượng."
            
        print(f"--> Feedback: {feedback}")

if __name__ == "__main__":
    main()