import torch
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 物理参数
    Lx: float = 2.0
    Ly: float = 1.0
    c: float = 0.15
    total_time: float = 20.0
    
    # 网络结构参数
    input_dim: int = 3
    hidden_dim: int = 128
    num_layers: int = 2
    activation_type: int = 5
    
    # 训练参数
    epochs: int = 2000
    batch_size: int = 2000
    
    # 可视化分辨率
    nx: int = 100
    ny: int = 100
    
    # 模型保存路径
    model_path: str = "trained_model.pth"
    
    # 设备配置 - 动态检测
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

settings = Settings()