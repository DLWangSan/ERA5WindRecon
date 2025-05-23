import torch
from torchinfo import summary

from models import STSRNetPlus

onnx_net_path = 'stsr_best.onnx'  # 设置onnx模型保存的权重路径
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = STSRNetPlus(t_scale=6, s_scale=2, extra_scale=2.5, c_in=7, use_coord=True).to(device)
checkpoint = torch.load("stsr_best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("✅ 模型加载完成")


input = torch.randn(8, 7, 6, 10, 8).to(device)
summary(model, input_size=(8, 7, 6, 10, 8), device=device)
torch.onnx.export(model, input, onnx_net_path, verbose=False)
