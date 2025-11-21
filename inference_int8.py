import torch
from torch.ao.quantization import prepare_qat, convert, get_default_qat_qconfig

from ultralytics.utils.ops import non_max_suppression
from ultralytics.qat.pytorch_native.qat_pytorch_trainer import QuantYolo

from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils.ops import scale_boxes, xyxy2xywh
from ultralytics.utils.plotting import Annotator
from ultralytics.data.augment import LetterBox

import os
import time
import cv2
import numpy as np
from tqdm import tqdm


def replace_silu_with_relu(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.SiLU):
            setattr(model, name, torch.nn.ReLU())
        else:
            replace_silu_with_relu(module)
            
def load_int8_model(ckpt_path: str, cfg_path: str, nc: int = 80):
    torch.backends.quantized.engine = "fbgemm"
    
    quantized_model = QuantYolo(cfg=cfg_path, ch=3, nc=nc)

    replace_silu_with_relu(quantized_model)

    quantized_model.fuse_model()
    quantized_model.qconfig = get_default_qat_qconfig("fbgemm")

    # QAT용 observer/quantizer 준비
    quantized_prepared = prepare_qat(quantized_model)
    
    # 실제 양자화 구조로 변환
    quantized = convert(quantized_prepared)

    # 저장해둔 state_dict 로드
    state_dict = torch.load(ckpt_path, map_location="cpu")
    quantized.load_state_dict(state_dict, strict=False)
    quantized.eval()

    print(f"✅ Quantized INT8 model loaded from state_dict ({torch.backends.quantized.engine}).")

    return quantized

def preprocess_image(image_path, img_size=960, stride=32, device='cpu'):
    img0 = cv2.imread(image_path)  # BGR
    assert img0 is not None, f"Image not found: {image_path}"

    transform = LetterBox(img_size, stride=stride)
    img = transform(image=img0)
    img = img[..., ::-1].transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0  # normalize
    img = img.unsqueeze(0).to(device)  # [1, 3, H, W]

    return img, img0

def run_inference(model, image_path, output_path, conf_thres=0.01, iou_thres=0.45, save_txt=True):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)

    img_tensor, original_img = preprocess_image(image_path, stride=int(model.stride.max()), device=device)
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        preds = model(img_tensor)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]    # YOLOv8 returns (inference, loss) tuple, so take [0]
        preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)[0]
    end_time = time.time()
    inference_time = (end_time - start_time) # second

    annotator = Annotator(original_img, line_width=3, font_size=2)

    # Create output folders
    img_filename = os.path.splitext(os.path.basename(image_path))[0]
    out_img_dir = os.path.join(output_path, 'images')
    out_txt_dir = os.path.join(output_path, 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_txt_dir, exist_ok=True)
    out_img_path = os.path.join(out_img_dir, img_filename + '.jpg')
    out_txt_path = os.path.join(out_txt_dir, img_filename + '.txt')

    # Draw boxes and save
    if preds is not None and len(preds):
        preds[:, :4] = scale_boxes(img_tensor.shape[2:], preds[:, :4], original_img.shape).round()
        for *xyxy, conf, cls in preds:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=(0, 0, 255), txt_color=(255, 255, 255))

        # Save YOLO-format txt
        if save_txt:
            with open(out_txt_path, 'w') as f:
                for *xyxy, conf, cls in preds:
                    xywh = xyxy2xywh(torch.tensor([xyxy]))[0]  # (x_center, y_center, w, h)
                    xywh /= torch.tensor(original_img.shape)[[1, 0, 1, 0]]  # normalize
                    f.write(f"{int(cls)} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {conf:.6f}\n")
    else:
        # Save empty label file if no detections
        if save_txt:
            with open(out_txt_path, 'w') as f:
                f.write(" ")

    # Save annotated result image
    result_img = annotator.result()
    cv2.imwrite(out_img_path, result_img)

    return inference_time


if __name__ == '__main__':
    ckpt_path = './weights/exports/best_int8_fixed.pt'
    model_cfg = './ultralytics/cfg/models/v8/yolov8s-p2.yaml'
    data_cfg = './dataset.yaml'
    input = './test_images'
    output_dir = './QAT_results'

    ckpt = torch.load(ckpt_path, map_location='cpu')
    data_yaml = yaml_model_load(data_cfg)
    model = load_int8_model(ckpt_path, model_cfg, nc=len(data_yaml['names']))

    for name, module in model.named_modules():
        if "quantized" in str(type(module)).lower():
            print(f"✅ Quantized layer detected: {name} -> {type(module)}")

    image_list = []
    for (root, dir, files) in os.walk(input):
        for file in files:
            image_list.append(os.path.join(root, file))
    print(f"Found {len(image_list)} images.")

    total_time = 0
    for image_path in tqdm(image_list):
        inference_time = run_inference(model, image_path, output_path=output_dir)
        total_time += inference_time

    print(f"평균 Inference 시간 : {total_time/len(image_list)} s")
