# train.py
import torch
import numpy as np
import time
from models.pinn_model import WavePINN
from data.data_generator import generate_data
from utils.visualization import plot_training_loss, visualize_sampling, plot_wave_field
from app.config import settings

# 损失函数计算
def compute_loss(pinn, ic_data, bc_data, pde_data):
    # 初始条件损失
    u_pred_ic = pinn(ic_data['inputs'])
    ic_loss = torch.mean((u_pred_ic - ic_data['exact'])**2) 
    
    # 边界条件损失
    u_pred_bc = pinn(bc_data['inputs'])
    bc_loss = torch.mean(u_pred_bc**2)
    
    # PDE残差计算
    inputs = pde_data['inputs'].clone().requires_grad_(True)
    u = pinn(inputs)
    
    # 计算二阶导数（输入顺序 [t, x, y]）
    gradients = torch.autograd.grad(
        outputs=u, inputs=inputs,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    u_t = gradients[:, 0]
    u_x = gradients[:, 1]
    u_y = gradients[:, 2]
    
    u_tt = torch.autograd.grad(
        outputs=u_t, inputs=inputs,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True, retain_graph=True
    )[0][:, 0]
    
    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=inputs,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 1]
    
    u_yy = torch.autograd.grad(
        outputs=u_y, inputs=inputs,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0][:, 2]
    
    pde_loss = torch.mean((u_tt - settings.c**2 * (u_xx + u_yy))**2)
    
    total_loss = 10.0 * pde_loss + ic_loss + bc_loss
    return total_loss, pde_loss, ic_loss, bc_loss

def test_model(pinn, test_times):
    """测试模型"""
    pinn.eval()
    with torch.no_grad():
        for t in test_times:
            x = torch.linspace(0, settings.Lx, settings.nx, dtype=torch.float64)
            y = torch.linspace(0, settings.Ly, settings.ny, dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            tt = torch.full_like(xx, t)
            inputs = torch.stack([tt.flatten(), xx.flatten(), yy.flatten()], dim=1).to(settings.device)
            u_pred = pinn(inputs).cpu().numpy().reshape(settings.nx, settings.ny)
            
            # 解析解
            u_exact = (np.sin(np.pi / settings.Lx * xx.numpy()) *
                      np.sin(np.pi / settings.Ly * yy.numpy()) *
                      np.cos(settings.c * np.sqrt((np.pi / settings.Lx)**2 + (np.pi / settings.Ly)**2) * t))
            
            # 计算 L2 误差
            l2_error = np.sqrt(np.mean((u_pred - u_exact)**2))
            print(f'L2 Error at t={t:.1f}: {l2_error:.2e}')
            
            # 可视化
            plot_wave_field(u_pred, xx.numpy(), yy.numpy(), t, l2_error)

if __name__ == "__main__":
    start_time = time.time()

    pinn = WavePINN().to(settings.device)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    
    best_loss = float('inf')
    losses = {
        'total': [],  # 总损失
        'pde': [],    # PDE损失
        'ic': [],     # 初始条件损失
        'bc': []      # 边界条件损失
    }

    for epoch in range(settings.epochs):
        data = generate_data()
        
        # 数据迁移到设备
        ic_data = {
            'inputs': data['ic']['inputs'].to(settings.device).requires_grad_(True),
            'exact': data['ic']['exact'].to(settings.device)
        }
        bc_data = {
            'inputs': data['bc']['inputs'].to(settings.device).requires_grad_(True)
        }
        pde_data = {
            'inputs': data['pde']['inputs'].to(settings.device).requires_grad_(True)
        }
        
        optimizer.zero_grad()
        total_loss, pde_loss, ic_loss, bc_loss = compute_loss(pinn, ic_data, bc_data, pde_data)
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        
        # 记录损失
        losses['total'].append(total_loss.item())
        losses['pde'].append(pde_loss.item())
        losses['ic'].append(ic_loss.item())
        losses['bc'].append(bc_loss.item())
        
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(pinn.state_dict(), settings.model_path)
            
        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:04d} | Loss: {total_loss.item():.2e} '
                  f'(PDE: {pde_loss.item():.2e}, IC: {ic_loss.item():.2e}, BC: {bc_loss.item():.2e}) '
                  f'LR: {lr:.1e}')
    
    # 绘制训练损失变化曲线
    plot_training_loss(losses)

    # 可视化采样点
    data = generate_data()
    visualize_sampling(data)

    # 使用L-BFGS微调（可选）
    print("Starting L-BFGS fine-tuning...")
    optimizer = torch.optim.LBFGS(pinn.parameters(), max_iter=1000, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        data = generate_data()
        ic_data = {
            'inputs': data['ic']['inputs'].to(settings.device).requires_grad_(True),
            'exact': data['ic']['exact'].to(settings.device)
        }
        bc_data = {
            'inputs': data['bc']['inputs'].to(settings.device).requires_grad_(True)
        }
        pde_data = {
            'inputs': data['pde']['inputs'].to(settings.device).requires_grad_(True)
        }
        total_loss, _, _, _ = compute_loss(pinn, ic_data, bc_data, pde_data)
        total_loss.backward()
        return total_loss
    optimizer.step(closure)

    # 测试模型
    test_model(pinn, [5, 10, 15, 20])

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds")