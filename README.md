# YOLOv8 QAT

   
I've made a general fix to the old code(https://github.com/mmsori/yolov8-QAT.git) that is not compatible with the current yolov8(ultralytics) repo.   
- Fixed an issue where class num('nc') was not overriding.   
- Modified to enable training with multi gpu(DDP).   
- Updated model saving logic
    - FP32 models are now saved as state_dict only.
    - INT8 (QAT-trained) models are saved as full model objects, preserving quantized modules correctly.
- INT8 inference workflow improved
    - The previously saved INT8 model could not be used directly for inference.
    - Added a conversion script that takes the saved FP32 model and converts it to an INT8 model using the QAT graph.
    - Added an inference script that performs detection using the newly converted INT8 model.
- Fixed several QAT-related bugs
    - Resolved multiple issues occurring during QAT training.   
      

## environment
ubuntu 22.04   
python 3.10  
pytorch 2.3.1   
cuda 12.1   
    
## Usage   
### Installation
```bash
python3.10 -m venv env-qat   

source env-qat/bin/activate   

pip install --upgrade pip   

pip install scikit-build cmake   

apt-get install git -y; apt install git   

apt-get install libgl1-mesa-glx -y   

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121    

```    

Install editable package in your environment by
```bash
pip install -r requirements.txt
```


### Using pytorch native quantization API   
```bash
python qat_pytorch.py --model_config yolov8n.yaml --pretrained_weight yolov8n.pt --data_config dataset/coco8.yaml \
--imgsz 640 --batch 8 --epochs 100 --device 0,1
```    
   
### inference
Conversion script that takes the saved FP32 model and converts it to an INT8 model using the QAT graph.   
```bash
python convert_fp2int.py
```   
   
Inference script that performs detection using the newly converted INT8 model.
```bash
python inference_int8.py
```   

## TODO
- end-to-end export to TensorRT engine(when using pytorch_quantization)
- yolov8 QAT using nvidia pytorch_quantization package
- code refactoring 

## References
https://github.com/mmsori/yolov8-QAT   

https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training   