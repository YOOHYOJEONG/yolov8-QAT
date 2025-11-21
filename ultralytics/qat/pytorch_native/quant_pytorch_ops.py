import os
import torch
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import QuantStub, DeQuantStub
from ultralytics.utils.tal import dist2bbox, make_anchors

# BACKEND = "qnnpack"
BACKEND = "fbgemm"
WORLD_SIZE = os.environ.get("WORLD_SIZE", 1)

def bottleneck_quant_forward(self, x):
    return self.ffn.add(x , self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))

def concat_quant_forward(self, x):
    return self.ffn.cat(x, self.d)

def c2f_quant_forward(self, x):
    """Forward pass through C2f layer."""
    x = self.cv1(x)
    x = self.dequant(x)
    y = list(x.split((self.c, self.c), 1)) # ONNX does not support quantized ops for split or chunk
    y = [self.quant(t) for t in y]
    y.extend(m(y[-1]) for m in self.m)
    
    return self.cv2(self.ffn.cat(y, 1))

def sppf_quant_forward(self,x):
    """Forward pass through Ghost Convolution block."""
    x = self.cv1(x)
    y1 = self.m(x)
    y2 = self.m(y1)
    return self.cv2(self.ffn.cat((x, y1, y2, self.m(y2)), 1))

def detect_quant_forward(self, x):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        x[i] = self.ffn.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    x = [self.dequant(ff) for ff in x]
    if self.training:
        return x
    #elif (self.dynamic or self.shape != shape):
    self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    self.shape = shape

    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    box = box.permute(0, 2, 1).contiguous()
    b, a, c =  box.shape  # batch, anchors, channels
    box = box.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(box.dtype).to(box.device)).permute(0,2,1)
    
    dbox = dist2bbox(box , self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

    y = torch.cat((dbox, self.sigmoid(cls)) , 1)
    return y if self.export else (y, x)

def quant_module_change(model):
    for name, module in model.named_modules():
        if module.__class__.__name__ == "C2f":
            module.__class__.forward = c2f_quant_forward
            module.dequant = DeQuantStub()
            module.quant = QuantStub()
            module.ffn = FloatFunctional()

        if module.__class__.__name__ == "Bottleneck":
            module.ffn = FloatFunctional()
            if module.add: 
                module.__class__.forward = bottleneck_quant_forward
                
        if module.__class__.__name__ == "Concat":
            module.ffn = FloatFunctional()
            module.__class__.forward = concat_quant_forward

        if module.__class__.__name__ == "SPPF":
            module.ffn = FloatFunctional()
            module.__class__.forward = sppf_quant_forward
        
        if module.__class__.__name__ == "Detect":
            module.sigmoid = torch.nn.Sigmoid()
            module.ffn = FloatFunctional()
            module.proj = torch.arange(16, dtype=torch.float)
            module.__class__.forward = detect_quant_forward