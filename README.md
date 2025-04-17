# YOLOv8 QAT

   
I've made a general fix to the old code(https://github.com/mmsori/yolov8-QAT.git) that is not compatible with the current yolov8(ultralytics) repo.   
- Fixed an issue where class num('nc') was not overriding.   
- Modified to enable training with multi gpu(DDP).   
- When saving the model, I modified it to save fp32 model as well. Since the int8 model saved in the previous version can only perform CPU operations, I also save the fp32 model that can perform GPU operations.   
- I wrote an inference code that can be inferred from the fp32 model I saved using GPU.   
(I used the ultralytics module because it was an ultralytics base.) 
      

## environment
ubuntu 22.04   
python 3.10  
pytorch 2.3.1   
cuda 12.1   
    
## Usage   
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
pip install -e .
```


### Using pytorch native quantization API   
```bash
python qat_pytorch.py --model_config yolov8n.yaml --pretrained_weight yolov8n.pt --data_config dataset/coco8.yaml --imgsz 640 --batch 8 --epochs 100 --device 0,1
```    
### Using `pytorch_quantization` package from nvidia
You need to install `pytorch_quantization` package   
```bash
# TODO
```

## TODO
- end-to-end export to TensorRT engine(when using pytorch_quantization)
- yolov8 QAT using nvidia pytorch_quantization package
- code refactoring 

## References
https://github.com/mmsori/yolov8-QAT   

https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training   