# models/pinn_model.py
import torch
import torch.nn as nn
import numpy as np
from app.config import settings

class Activation(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_number_x = np.pi / settings.Lx
        self.wave_number_y = np.pi / settings.Ly
        self.c = settings.c
        
    def forward(self, x, original_input):
        t_coord = original_input[:, 0]
        x_coord = original_input[:, 1]
        y_coord = original_input[:, 2]
        
        wave_number_x_tensor = torch.tensor(self.wave_number_x, dtype=torch.float64, device=x.device)
        wave_number_y_tensor = torch.tensor(self.wave_number_y, dtype=torch.float64, device=x.device)
        omega = self.c * torch.sqrt(wave_number_x_tensor**2 + wave_number_y_tensor**2)
        
        if settings.activation_type == 1:
            return torch.relu(x)
        elif settings.activation_type == 2:
            return torch.sigmoid(x)
        elif settings.activation_type == 3:
            return torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        elif settings.activation_type == 4:
            return x * torch.sigmoid(x)
        elif settings.activation_type == 5:
            wave = (torch.sin(wave_number_x_tensor * x_coord) *
                   torch.sin(wave_number_y_tensor * y_coord) *
                   torch.cos(omega * t_coord))
            return 0.5 * (wave.unsqueeze(1) + torch.tanh(x))
        elif settings.activation_type == 6:
            return torch.sin(x)
        else:
            return torch.tanh(x)

class WavePINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for i in range(settings.num_layers-1):
            layers.append(nn.Linear(settings.input_dim if i==0 else settings.hidden_dim, 
                                    settings.hidden_dim, 
                                    dtype=torch.float64))
            layers.append(Activation())
        layers.append(nn.Linear(settings.hidden_dim, 1, dtype=torch.float64))
        self.net = nn.Sequential(*layers)
        
        # 参数初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, mean=0.0, std=0.1)
    
    def forward(self, inputs):
        # 移除显式设备转换，由调用方处理
        x = inputs
        for layer in self.net:
            if isinstance(layer, Activation):
                x = layer(x, inputs)  # 使用原始输入
            else:
                x = layer(x)
        return x