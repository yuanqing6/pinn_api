# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from app.config import settings

def plot_training_loss(losses):
    """绘制训练损失曲线"""
    total_loss_calculated = [
        10.0 * pde + ic + bc for pde, ic, bc in zip(losses['pde'], losses['ic'], losses['bc'])
    ]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(total_loss_calculated) + 1), total_loss_calculated, 
               label='Total Loss (Weighted)', color='blue')
    plt.loglog(range(1, len(losses['pde']) + 1), losses['pde'], 
               label='PDE Loss', color='orange')
    plt.loglog(range(1, len(losses['ic']) + 1), losses['ic'], 
               label='Initial Condition Loss', color='green')
    plt.loglog(range(1, len(losses['bc']) + 1), losses['bc'], 
               label='Boundary Condition Loss', color='red')
    plt.xlabel('Epoch (log scale)')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss Curve (Log-Log)')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.savefig('training_loss.png')
    plt.show()

def visualize_sampling(data):
    """可视化采样点分布"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data['ic']['inputs'][:, 1].cpu().numpy(), 
                data['ic']['inputs'][:, 2].cpu().numpy(), 
                label='Initial Condition', alpha=0.6, s=10)
    plt.scatter(data['bc']['inputs'][:, 1].cpu().numpy(), 
                data['bc']['inputs'][:, 2].cpu().numpy(), 
                label='Boundary Condition', alpha=0.6, s=10)
    plt.scatter(data['pde']['inputs'][:, 1].cpu().numpy(), 
                data['pde']['inputs'][:, 2].cpu().numpy(), 
                label='PDE Residual', alpha=0.6, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sampling Points Visualization')
    plt.legend()
    plt.grid()
    plt.show()

def plot_wave_field(u_pred, xx, yy, t, l2_error=None):
    """绘制波场分布"""
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(xx, yy, u_pred, levels=50, cmap='viridis')
    plt.colorbar(contour)
    title = f'Wave Field at t={t:.1f}'
    if l2_error is not None:
        title += f' (L2 Error: {l2_error:.2e})'
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()