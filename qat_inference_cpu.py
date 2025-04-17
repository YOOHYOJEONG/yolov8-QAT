import torch
from ultralytics.qat.pytorch_native.qat_pytorch_trainer import QuantYolo
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.ops import scale_boxes, xyxy2xywh
from ultralytics.data.augment import LetterBox


import os
import time
import cv2
import numpy as np
from tqdm import tqdm


def load_int8_model(ckpt_path: str, cfg_path: str, nc: int = 80, device: str = 'cpu'):
    # 모델 구조 정의
    model = QuantYolo(cfg=cfg_path, ch=3, nc=nc)
    
    # checkpoint 로드
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # INT8로 convert 된 모델은 state_dict만 사용
    model.load_state_dict(ckpt['model'], strict=False)

    # CPU에 올리고 eval 모드
    model.to(device).eval()
    return model

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
    from ultralytics.utils.ops import non_max_suppression

    img_tensor, original_img = preprocess_image(image_path, stride=int(model.stride.max()), device=next(model.parameters()).device)
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        preds = model(img_tensor)[0]  # YOLOv8 returns (inference, loss) tuple, so take [0]
        preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)[0]
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # milliseconds

    annotator = Annotator(original_img, line_width=3, font_size=2)

    # Save output image
    os.makedirs(output_path, exist_ok=True)
    img_filename = os.path.splitext(os.path.basename(image_path))[0]
    out_img_path = os.path.join(output_path, 'images', img_filename + '.jpg')
    out_txt_path = os.path.join(output_path, 'labels', img_filename + '.txt')
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)

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
        if save_txt:
            with open(out_txt_path, 'w') as f:
                f.write(" ")
                    
    cv2.imwrite(out_img_path, annotator.result())
    return inference_time


if __name__ == '__main__':
    ckpt_path = 'weights/best_fp32.pt'
    model_cfg = 'yolov8n.yaml'
    data_cfg = 'coco8.yaml'
    input = 'test/images'
    output_dir = 'test/results'

    ckpt = torch.load(ckpt_path, map_location='cpu')
    data_yaml = yaml_model_load(data_cfg)
    model = load_int8_model(ckpt_path, model_cfg, nc=len(data_yaml['names']))

    image_list = []
    for (root, dir, files) in os.walk(input):
        for file in files:
            image_list.append(os.path.join(root, file))
    print(f"Found {len(image_list)} images.")

    total_time = 0
    for image_path in tqdm(image_list):
        inference_time = run_inference(model, image_path, output_path=output_dir)
        total_time += inference_time

    print(f"평균 Inference 시간 : {total_time/len(image_list)}")
