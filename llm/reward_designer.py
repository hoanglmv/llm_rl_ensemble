# llm/reward_designer.py
import random

class LLMRewardDesigner:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.iteration = 0

    def generate_code(self, feedback_report):
        """
        Trong thực tế, hàm này sẽ gọi OpenAI/Gemini API với prompt engineering.
        Ở đây, tôi mô phỏng sự 'tiến hóa' của LLM qua 3 giai đoạn để bạn chạy thử.
        """
        self.iteration += 1
        print(f"\n[LLM Architect] Đang phân tích feedback và viết lại code (Iter {self.iteration})...")
        
        # Kịch bản mô phỏng tiến hóa dựa trên bài báo
        if self.iteration == 1:
            # Giai đoạn 1: Code ngây thơ (chỉ quan tâm năng lượng)
            code = "reward = -power"
            comment = "# V1: Chỉ tối ưu năng lượng (sẽ gây rớt mạng nhiều)"
            
        elif self.iteration == 2:
            # Giai đoạn 2: Bắt đầu phạt drop rate (nhưng chưa đủ mạnh)
            code = "reward = -power - 1000 * drop_rate"
            comment = "# V2: Phạt drop_rate nhưng trọng số thấp"
            
        else:
            # Giai đoạn 3: Tối ưu cân bằng (Phạt cực nặng drop_rate + switching)
            code = "reward = -power - 50000 * drop_rate - 20 * switches"
            comment = "# V3: Tối ưu cân bằng (theo Fig.12 bài báo)"
            
        full_response = f"{comment}\n{code}"
        return full_response