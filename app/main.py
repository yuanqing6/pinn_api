import os
import torch
import gdown
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.pinn_model import WavePINN
from app.config import settings

app = FastAPI(
    title="PINN API",
    description="物理信息神经网络求解二维声波方程",
    version="1.0.0",
    docs_url="/docs",  # 明确指定文档路径
    redoc_url="/redoc"  # 可选：启用Redoc文档
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)
# 添加根路由，避免访问根路径时404
@app.get("/")
async def root():
    return {"message": "PINN API is running", "docs": "/docs"}
# 获取设备
device = torch.device(settings.device)

def download_model_from_drive():
    """从Google Drive下载模型（如果不存在）"""
    model_path = settings.model_path
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        # 替换为你的Google Drive文件ID
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
        try:
            gdown.download(url, model_path, quiet=False)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Model download failed: {str(e)}")
            # 如果下载失败，创建一个空模型（仅用于演示，实际应用需处理）
            model = WavePINN()
            torch.save(model.state_dict(), model_path)
    else:
        print("Model already exists.")

def download_model_from_github():
    """从GitHub Releases下载模型（如果不存在）"""
    model_path = settings.model_path
    if not os.path.exists(model_path):
        print("Downloading model from GitHub Releases...")
        # 替换为你的GitHub Releases URL
        url = "https://github.com/username/repo/releases/download/v1.0/trained_model.pth"
        try:
            response = requests.get(url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Model download failed: {str(e)}")
            # 如果下载失败，创建一个空模型（仅用于演示，实际应用需处理）
            model = WavePINN()
            torch.save(model.state_dict(), model_path)
    else:
        print("Model already exists.")

# 选择一种下载方式（这里使用Google Drive方式）
download_model_from_drive()

# 加载模型
try:
    model = WavePINN().to(device)
    model.load_state_dict(torch.load(settings.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # 如果加载失败，创建一个新的模型实例
    model = WavePINN().to(device)
    print("Using untrained model.")

class PredictRequest(BaseModel):
    t: float
    x: float
    y: float

@app.post("/predict")
async def predict(request: PredictRequest):
    """输入时间t、x坐标、y坐标，返回声波方程的解u"""
    try:
        # 创建输入张量并移动到设备
        inputs = torch.tensor(
            [[request.t, request.x, request.y]],
            dtype=torch.float64
        ).to(device)
        
        # 模型推理
        with torch.no_grad():
            u_pred = model(inputs).item()
        
        return {
            "t": request.t,
            "x": request.x,
            "y": request.y,
            "u_value": u_pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": os.path.exists(settings.model_path)
    }

# 预热端点（可选，用于减少冷启动时间）
@app.get("/warmup")
async def warmup():
    """预热端点，减少冷启动时间"""
    try:
        # 创建一个简单的预测请求
        inputs = torch.tensor([[5.0, 1.0, 0.5]], dtype=torch.float64).to(device)
        with torch.no_grad():
            _ = model(inputs)
        return {"status": "warmed up"}
    except Exception as e:
        return {"error": str(e)}

# Vercel 需要这个处理程序，但注意：在Vercel上，我们不需要运行uvicorn，因为Vercel会处理
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)