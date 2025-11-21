import os
import sys
import argparse
import subprocess
import textwrap
import json
import socket

from ultralytics.utils import LOGGER, colorstr


def find_free_network_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def generate_ddp_file(overrides):
    import tempfile

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    
    overrides_json = json.dumps(overrides)

    helper_definitions = textwrap.dedent(f"""
        import os
        import json
        from ultralytics.qat.pytorch_native.qat_pytorch_trainer import PytorchQuantizationTrainer
        from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, get_default_qat_qconfig, convert, disable_observer, fuse_modules_qat
        from torch.nn.intrinsic.qat import freeze_bn_stats

        import time
        import math
        import numpy as np
        from copy import deepcopy
        import warnings
        from datetime import datetime

        import torch 
        import torch.nn as nn
        from torch import nn, optim
        from torch import distributed as dist

        from ultralytics.models.yolo.detect.train import DetectionTrainer
        from ultralytics.nn.tasks import DetectionModel, BaseModel, v8DetectionLoss,parse_model, yaml_model_load, attempt_load_weights
        from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, __version__, callbacks, TQDM, colorstr, emojis
        from ultralytics.utils.torch_utils import  EarlyStopping, ModelEMA, initialize_weights, scale_img
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.head import Detect, Segment, Pose
        from ultralytics.utils.autobatch import check_train_batch_size
        from ultralytics.utils.checks import check_amp, check_imgsz
        from ultralytics.qat.pytorch_native.qat_pytorch_trainer import QuantYolo, QATWrappedModel, load_quantized_model

        from ultralytics.qat.pytorch_native.quant_pytorch_ops import quant_module_change, BACKEND, WORLD_SIZE
    """)

    runner_code = textwrap.dedent(f"""
        if __name__ == '__main__':
            overrides = json.loads('''{overrides_json}''')
            trainer = PytorchQuantizationTrainer(cfg=DEFAULT_CFG, overrides=overrides)
            trainer.model = trainer.get_model(cfg=overrides['cfg'],  # model cfg
                                          weights=overrides['model'],  # pretrained_weight
                                          backend=overrides['quant_backend'])
            trainer.train()
    """)

    temp_file.write(helper_definitions + "\n" + runner_code)
    temp_file.close()
    return temp_file.name


def generate_ddp_command(world_size, overrides):
    file = generate_ddp_file(overrides)
    port = find_free_network_port()
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(world_size),
        "--master_port", str(port),
        file
    ]
    return cmd, file


def ddp_cleanup(file):
    if os.path.exists(file):
        os.remove(file)


def train(args):
    if isinstance(args.device, str):
        device_ids = [int(d) for d in args.device.split(',') if d.strip().isdigit()]
        world_size = len(device_ids)
    else:
        world_size = 1

    # Build cfg from args
    overrides = {
        'cfg': args.model_config,
        'model': args.pretrained_weight,
        'data': args.data_config,
        'imgsz': args.imgsz,
        'epochs': args.epochs,
        'device': args.device,
        'batch': args.batch,
        'quant_backend': args.quant_backend,
        'project': args.project,
        'name': args.name,
    }

    # DDP
    if world_size > 1 and 'LOCAL_RANK' not in os.environ:
        cmd, temp_file = generate_ddp_command(world_size, overrides)
        LOGGER.info(f"{colorstr('DDP:')} launching distributed training via: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        ddp_cleanup(temp_file)
        return

    # single gpu
    from ultralytics.qat.pytorch_native.qat_pytorch_trainer import PytorchQuantizationTrainer
    from ultralytics.utils import DEFAULT_CFG

    print("Starting QAT training on single GPU/CPU...")
    trainer = PytorchQuantizationTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer.model = trainer.get_model(
        cfg=overrides['cfg'],
        weights=overrides['model'],
        backend=overrides['quant_backend']
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default='yolov8s.yaml')
    parser.add_argument("--pretrained_weight", type=str, default='yolov8s.pt')
    parser.add_argument("--data_config", type=str, default='coco.yaml')
    parser.add_argument("--quant_backend", type=str, default='qnnpack')
    parser.add_argument("--imgsz", type=int, default=640, help="input images size as int for train and val modes, or list[h,w] for predict and export modes")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs, It must always be a multiple of 10")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument('--batch', type=int, default=4, help="Batch-size for QAT")
    parser.add_argument('--project', type=str, default="runs/train/yolov8", help="project name")
    parser.add_argument('--name', type=str, default="test", help="experiment name, results saved to 'project/name' directory")
    args = parser.parse_args()

    train(args)