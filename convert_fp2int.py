import torch
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.qat.pytorch_native.qat_pytorch_trainer import QuantYolo
from torch.ao.quantization import prepare_qat, convert, get_default_qat_qconfig


def replace_silu_with_relu(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.SiLU):
            setattr(model, name, torch.nn.ReLU())
        else:
            replace_silu_with_relu(module)

data_cfg = './dataset.yaml'
ckpt = torch.load(
    "./weights/best_fp32.pt",
    map_location="cpu"
)
data_yaml = yaml_model_load(data_cfg)

model = QuantYolo(
    cfg="./ultralytics/cfg/models/v8/yolov8s-p2.yaml",
    ch=3,
    nc=len(data_yaml['names'])
)

replace_silu_with_relu(model)

model.fuse_model()
model.qconfig = get_default_qat_qconfig("qnnpack")
model_prepared = prepare_qat(model)
model_prepared.load_state_dict(ckpt["model"], strict=False)
model_prepared.eval()

# enable_observer / disable_observer는 observer 객체가 아닌 모듈에서 호출해야 함
def try_enable_disable_observer(m, enable=True):
    if hasattr(m, "observer_enabled"):
        m.observer_enabled[0] = enable
    elif hasattr(m, "fake_quant_enabled"):
        m.fake_quant_enabled[0] = enable

# observer 활성화
for m in model_prepared.modules():
    try_enable_disable_observer(m, enable=True)

# dummy forward (입력 한두 장 통과시키기)
print("Running dummy forward pass to initialize quantization parameters...")
x = torch.randn(1, 3, 640, 640)
with torch.inference_mode():
    _ = model_prepared(x)

# observer 비활성화 (freeze)
for m in model_prepared.modules():
    try_enable_disable_observer(m, enable=False)

quantized_model = convert(model_prepared)
torch.save(
    quantized_model.state_dict(),
    "./weights/exports/best_int8_fixed.pt"
)

print("✅ INT8 model successfully converted and saved.")