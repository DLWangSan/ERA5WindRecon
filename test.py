import torch
from torchviz import make_dot
from models import STSRNetPlus

# 构造模型
model = STSRNetPlus(t_scale=6, s_scale=2, extra_scale=2.5, c_in=7, use_coord=True, use_lsm=True)
model.eval()

# 构造输入（batch size=1 避免图过大）
dummy_input = torch.randn(1, 7, 6, 10, 8)
output = model(dummy_input)

# 生成图
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("stsrnetplus_graph", format="png")  # 生成 stsrnetplus_graph.png
