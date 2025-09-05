import os
import torch
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import gdown
import requests

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PINN API",
    description="物理信息神经网络求解二维声波方程",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 模型定义
class WavePINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(3, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.output = torch.nn.Linear(64, 1)
        self.activation = torch.nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.output(x)

# 加载模型函数
def load_model():
    model_path = "trained_model.pth"
    
    # 如果模型文件不存在，从 GitHub Releases 下载
    if not os.path.exists(model_path):
        logger.info("Downloading model from GitHub Releases...")
        try:
            # 替换为您的实际 URL
            url = "https://github.com/yuanqing6/pinn-api/releases/download/v1.0/trained_model.pth"
            response = requests.get(url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            logger.info("Model downloaded successfully from GitHub Releases")
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            # 创建空模型作为后备
            model = WavePINN()
            torch.save(model.state_dict(), model_path)
            return model
    
    # 加载模型
    try:
        model = WavePINN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        # 创建空模型作为后备
        model = WavePINN()
        torch.save(model.state_dict(), model_path)
        return model

# 全局模型实例
pinn = load_model()

class PredictRequest(BaseModel):
    t: float
    x: float
    y: float

@app.post("/predict")
async def predict(request: PredictRequest):
    """输入时间t、x坐标、y坐标，返回声波方程的解u"""
    try:
        # 创建张量并移动到正确设备
        inputs = torch.tensor(
            [[request.t, request.x, request.y]], 
            dtype=torch.float32
        ).to(device)
        
        with torch.no_grad():
            u_pred = pinn(inputs).item()
        
        return {
            "t": request.t,
            "x": request.x,
            "y": request.y,
            "u_value": u_pred
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "device": str(device)}

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "PINN API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

# 预热端点（可选）
@app.get("/warmup")
async def warmup():
    """预热模型端点"""
    try:
        inputs = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32).to(device)
        with torch.no_grad():
            _ = pinn(inputs)
        return {"status": "warmed up"}
    except Exception as e:
        return {"error": str(e)}
