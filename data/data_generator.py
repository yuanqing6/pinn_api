# data/data_generator.py
import torch
import numpy as np
from app.config import settings

def generate_data(batch_size=settings.batch_size, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    # 初始条件数据 (t=0)
    x_ic = torch.rand(batch_size, dtype=torch.float64) * settings.Lx
    y_ic = torch.rand(batch_size, dtype=torch.float64) * settings.Ly
    t_ic = torch.zeros(batch_size, dtype=torch.float64)
    inputs_ic = torch.stack([t_ic, x_ic, y_ic], dim=1)
    exact_ic = (torch.sin(np.pi/settings.Lx * x_ic) * torch.sin(np.pi/settings.Ly * y_ic)).unsqueeze(1)
    
    # 边界条件数据 (四边)
    t_bc = torch.rand(4*batch_size//4, dtype=torch.float64) * settings.total_time
    
    # 左边界 (x=0)
    x_left = torch.zeros(batch_size//4, dtype=torch.float64)
    y_left = torch.rand(batch_size//4, dtype=torch.float64) * settings.Ly
    
    # 右边界 (x=Lx)
    x_right = torch.full((batch_size//4,), settings.Lx, dtype=torch.float64)
    y_right = torch.rand(batch_size//4, dtype=torch.float64) * settings.Ly
    
    # 下边界 (y=0)
    x_bottom = torch.rand(batch_size//4, dtype=torch.float64) * settings.Lx
    y_bottom = torch.zeros(batch_size//4, dtype=torch.float64)
    
    # 上边界 (y=Ly)
    x_top = torch.rand(batch_size//4, dtype=torch.float64) * settings.Lx
    y_top = torch.full((batch_size//4,), settings.Ly, dtype=torch.float64)
    
    x_bc = torch.cat([x_left, x_right, x_bottom, x_top])
    y_bc = torch.cat([y_left, y_right, y_bottom, y_top])
    
    inputs_bc = torch.stack([t_bc, x_bc, y_bc], dim=1)
    
    # PDE数据（内部点）
    t_pde = torch.rand(batch_size, dtype=torch.float64) * settings.total_time
    x_pde = torch.rand(batch_size, dtype=torch.float64) * settings.Lx
    y_pde = torch.rand(batch_size, dtype=torch.float64) * settings.Ly
    inputs_pde = torch.stack([t_pde, x_pde, y_pde], dim=1)
    
    return {
        'ic': {'inputs': inputs_ic, 'exact': exact_ic},
        'bc': {'inputs': inputs_bc},
        'pde': {'inputs': inputs_pde}
    }