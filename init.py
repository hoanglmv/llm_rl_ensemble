import os
from pathlib import Path

def create_project_structure():
    # Tên thư mục gốc
    root_dir = "AutoEnergySaving"

    # Danh sách các file cần tạo kèm theo đường dẫn tương đối
    files_to_create = [
        "configs/settings.py",
        "envs/__init__.py",
        "envs/telecom_env.py",
        "llm/__init__.py",
        "llm/reward_designer.py",
        "agents/__init__.py",
        "agents/ppo_agent.py",
        "main.py",
        "requirements.txt"
    ]

    # Tạo thư mục gốc
    base_path = Path(root_dir)
    if not base_path.exists():
        base_path.mkdir()
        print(f"Đã tạo thư mục gốc: {root_dir}")
    else:
        print(f"Thư mục {root_dir} đã tồn tại.")

    # Tạo các file và thư mục con
    for file_path in files_to_create:
        full_path = base_path / file_path
        
        # Tạo thư mục cha nếu chưa tồn tại (ví dụ: configs, envs...)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo file rỗng
        if not full_path.exists():
            full_path.touch()
            print(f"✅ Đã tạo file: {full_path}")
        else:
            print(f"⚠️ File đã tồn tại: {full_path}")

    print("\nHoàn tất khởi tạo project!")

if __name__ == "__main__":
    create_project_structure()