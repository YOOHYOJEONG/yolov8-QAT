import torch
from ultralytics.qat.pytorch_native.qat_pytorch_trainer import QuantYolo
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.ops import scale_boxes
from ultralytics.data.augment import LetterBox

import cv2
import numpy as np


def load_fp32_model(ckpt_path: str, cfg_path: str, nc: int = 80, device: str = 'cuda'):
    # 1. 모델 구조 정의 및 초기화
    model = QuantYolo(cfg=cfg_path, ch=3, nc=nc)
    
    # 2. 저장된 state_dict 로드
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)

    # 3. GPU에 올리고 eval 모드
    model.to(device).eval()
    return model

def preprocess_image(image_path, img_size=960, stride=32, device='cuda'):
    # 이미지 로드 및 전처리 (letterbox)
    img0 = cv2.imread(image_path)  # BGR
    assert img0 is not None, f"Image not found: {image_path}"

    transform = LetterBox(img_size, stride=stride)
    img = transform(image=img0)
    img = img[..., ::-1].transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0  # normalize
    img = img.unsqueeze(0).to(device)  # [1, 3, H, W]
    return img, img0

def run_inference(model, image_path, conf_thres=0.1, iou_thres=0.45):
    from ultralytics.utils.ops import non_max_suppression

    img_tensor, original_img = preprocess_image(image_path, stride=int(model.stride.max()), device=next(model.parameters()).device)
    
    # Forward pass
    with torch.no_grad():
        preds = model(img_tensor)[0]  # YOLOv8 returns (inference, loss) tuple, so take [0]
        preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)[0]

    print(preds)
    # 결과 디코딩 및 시각화
    annotator = Annotator(original_img, line_width=3, font_size=2)
    if preds is not None and len(preds):
        preds[:, :4] = scale_boxes(img_tensor.shape[2:], preds[:, :4], original_img.shape).round()
        for *xyxy, conf, cls in preds:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=(0, 0, 255), txt_color=(255,255,255))
    return annotator.result()


if __name__ == '__main__':
    # 설정값
    ckpt_path = 'yolov8/best_fp32.pt'
    model_cfg = 'yolov8n.yaml'
    data_cfg = 'coco8.yaml'
    test_image = 'assets/bus.jpg'

    # 모델 로드
    ckpt = torch.load(ckpt_path, map_location='cpu')
    data_yaml = yaml_model_load(data_cfg)
    model = load_fp32_model(ckpt_path, model_cfg, nc=len(data_yaml['names']))

    # 추론 실행
    result_img = run_inference(model, test_image)

    # 결과 출력
    cv2.imwrite("/assets/bus_infer.jpg", result_img)