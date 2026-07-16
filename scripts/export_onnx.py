import torch
import sys
sys.path.append('/home/rashadulnafisriyad/FYP_UNet/LWEU_NET/src/models/lweunet')
from lweunet_v2 import LWEUNetV2

model = LWEUNetV2()
checkpoint = torch.load(
    'checkpoints/lweunet_v2/best_model.pth',
    map_location='cpu'
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

dummy_input = torch.randn(1, 1, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'enhunet_v2.onnx',
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    verbose=False
)
print("ONNX export successful")







