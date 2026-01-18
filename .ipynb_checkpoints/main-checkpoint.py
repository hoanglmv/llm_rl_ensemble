# main.py
import hydra
import os
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from envs.telecom_env import TelecomEnv
from utils.read import load_dataset
from llm.reward_designer import LLMRewardDesigner
from agents.ppo_agent import DRLAgent

# Thêm tham số dataset_name vào config khi chạy
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Xác định tên dataset (từ command line hoặc config mặc định)
    # Ví dụ: python main.py dataset_name="data_C5_S24_U50"
    if "dataset_name" not in cfg:
        # Nếu không nhập, thử tự đoán tên mặc định dựa trên config hiện tại
        dataset_name = f"data_C{cfg.network.num_cells}_S{cfg.traffic.simulation_steps}_U{cfg.traffic.max_users}"
    else:
        dataset_name = cfg.dataset_name

    print(f"=== Đang chạy với Dataset: {dataset_name} ===")
    
    # 2. Load Data
    try:
        data_pack = load_dataset(dataset_name)
    except FileNotFoundError:
        print("❌ Lỗi: Chưa tạo dataset. Hãy chạy 'python utils/create.py' trước!")
        return

    # 3. Khởi tạo
    env = TelecomEnv(cfg, data_pack) # Truyền data vào env
    llm = LLMRewardDesigner()
    agent = DRLAgent(env, cfg)
    
    history_power = []
    history_drop = []
    
    # 4. Vòng lặp Tiến hóa
    rounds = cfg.llm.simulation_rounds
    feedback = "Khởi đầu."
    
    for i in range(rounds):
        print(f"\n--- ROUND {i+1} ---")
        reward_code = llm.generate_code(feedback)
        print(f"Reward: {reward_code}")
        env.reward_function_code = reward_code
        
        agent.train()
        metrics = agent.evaluate(episodes=5)
        
        p, d = metrics['avg_power'], metrics['avg_drop_rate']
        history_power.append(p)
        history_drop.append(d)
        
        print(f"Result: Power={p:.1f}, Drop={d*100:.2f}%")
        
        if d > cfg.rl.threshold_drop:
            feedback = f"BAD. Drop Rate {d:.2f} > {cfg.rl.threshold_drop}. Reduce drop rate!"
        else:
            feedback = "GOOD. Focus on saving power."

    # 5. Vẽ và Lưu biểu đồ (Figures)
    save_fig_dir = os.path.join(hydra.utils.get_original_cwd(), "figures")
    os.makedirs(save_fig_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_power, marker='o', color='b')
    plt.title("Average Power Consumption")
    plt.xlabel("Round")
    plt.ylabel("Watts")
    
    plt.subplot(1, 2, 2)
    plt.plot(np.array(history_drop)*100, marker='s', color='r')
    plt.title("Average Drop Rate")
    plt.xlabel("Round")
    plt.ylabel("Drop Rate (%)")
    
    fig_name = f"result_{dataset_name}.png"
    plt.savefig(os.path.join(save_fig_dir, fig_name))
    print(f"\n📊 Đã lưu biểu đồ kết quả tại: figures/{fig_name}")

if __name__ == "__main__":
    main()