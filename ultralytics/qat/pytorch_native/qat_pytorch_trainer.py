from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, get_default_qat_qconfig, convert, enable_observer, disable_observer, fuse_modules_qat
from torch.ao.nn.intrinsic.qat import freeze_bn_stats

import time
import math
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import gc

import torch 
import torch.nn as nn
from torch import nn, optim
from torch import distributed as dist

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, BaseModel, v8DetectionLoss, parse_model, yaml_model_load, attempt_load_weights
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, LOCAL_RANK, __version__, callbacks, TQDM, colorstr
from ultralytics.utils.torch_utils import  EarlyStopping, ModelEMA, initialize_weights, scale_img
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment, Pose
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_imgsz

from ultralytics.qat.pytorch_native.quant_pytorch_ops import quant_module_change, BACKEND, WORLD_SIZE


# QAT 후 convert()된 모델을 validator와 호환되도록 감싸는 래퍼 클래스
class QATWrappedModel(nn.Module):
    def __init__(self, quant_model, names, stride):
        super().__init__()
        self.model = quant_model
        self.names = names
        if isinstance(stride, torch.Tensor):
            self.stride = stride
        else:
            self.stride = torch.tensor([stride])
        self.pt = False
        self.amp = False
        self.nc = len(names)
        self.end2end = False

    def forward(self, x, augment=False, visualize=False, embed=None):
        return self.model(x)

    def loss(self, batch, preds):
        # validator에서 loss 계산을 위해 필요하므로 더미 리턴
        return torch.tensor(0.0), torch.zeros(3, device=preds.device)
    
    def fuse(self, verbose=False):  # validator(AutoBackend)가 호출하는 함수
        return self

def replace_silu_with_relu(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.SiLU):
            setattr(model, name, torch.nn.ReLU())
        else:
            replace_silu_with_relu(module)

def load_quantized_model(path, *_):
    model = torch.load(path, map_location='cpu')
    model.eval()
    wrapped_model = QATWrappedModel(model, getattr(model, 'names', {}), stride=getattr(model, 'stride', torch.tensor([32])))
    print("Quantized model successfully loaded and converted.")
    return wrapped_model

class QuantYolo(BaseModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=False):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)
        quant_module_change(self.model)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.model[-1].dequant = self.dequant

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        x = x.contiguous()  # CUDA kernel alignment 방지용
        x = self.quant(x)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x
    def _predict_augment(self, x):
            """Perform augmentations on input image x and return augmented inference and train outputs."""
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = super().predict(xi)[0]  # forward
                yi = self._descale_pred(yi, fi, si, img_size)
                y.append(yi)
            y = self._clip_augmented(y)  # clip augmented tails
            return torch.cat(y, -1), None  # augmented inference, train
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == Conv:
                if hasattr(m, 'bn') and m.bn is not None:
                    fuse_modules_qat(m, ['conv', 'bn', 'act'], inplace=True)

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)

class PytorchQuantizationTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.qat = 'native'

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}
       
    def get_model(self, cfg=None, weights=None, verbose=True, backend=BACKEND):
        Conv.default_act = nn.ReLU()
        self.model_cfg = cfg

        # 모델 생성
        model = QuantYolo(cfg=cfg, ch=3, nc=self.data['nc'], verbose=False)

        # 가중치 로드
        if weights:
            state = torch.load(weights, map_location='cpu')
            model.load_state_dict(state['model'].state_dict(), strict=False)

        # SiLU → ReLU 교체 (QAT 호환)
        replace_silu_with_relu(model)
        print("🔄 SiLU activations replaced with ReLU for QAT compatibility.")

        # fuse
        model.fuse_model()

        # QAT 설정
        model.qconfig = get_default_qat_qconfig(backend)  # 'fbgemm'
        prepare_qat(model, inplace=True)

        # observer warm-up (QAT 준비 직후 observer 초기화용)
        model.train()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            model(dummy)

        num_fakequants = sum(isinstance(m, torch.ao.quantization.FakeQuantize) for m in model.modules())
        print(f"🔍 FakeQuant modules found: {num_fakequants}")

        return model
    
    def setup_model(self):

        return None

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= self.epochs * 3.0, eta_min=self.args.lr0 * 0.01)

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        self.args.warmup_epochs = 0
        self.args.lr0 /= 10
        self.args.optimizer = 'SGD'

        # Model 준비
        self.run_callbacks('on_pretrain_routine_start')

        ckpt = self.setup_model()
        self.model = self.model.to(self.device)

        self.set_model_attributes()

        # Freeze layers 설정
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(False).to(self.device)  # Must be FALSE in QAT setting
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')

        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations)
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

    def _do_train(self, world_size=WORLD_SIZE):
        """Main training loop for QAT."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks('on_train_start')

        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for '
                    f'{self.args.time} hours...' if self.args.time else f'{self.epochs} epochs...')
        
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # 50% 시점: Observer freeze
            if epoch == int(self.epochs * 0.5):
                print("Disabling observers (fix quantization scales)")
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                self.model.apply(disable_observer)
                gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
                self.model.to(self.device)
                if dist.is_initialized():
                    dist.barrier()

            # 80% 시점: BN freeze
            if epoch == int(self.epochs * 0.8):
                print("Freezing BatchNorm running stats")
                self.model.apply(freeze_bn_stats)
                self.model.to(self.device)
                if dist.is_initialized():
                    dist.barrier()

            # 에폭 시작
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            nb = len(self.train_loader)
            pbar = enumerate(self.train_loader)

            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)

            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.amp.autocast(device_type='cuda', enabled=self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks('on_train_epoch_end')
            
            if RANK in (-1, 0):
                log_qat_observer_status(self.model, step=epoch)

                final_epoch = epoch + 1 == self.epochs
                # self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness)
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    if RANK in (-1, 0):
                        self.save_model()
                        self.run_callbacks('on_model_save')

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.scheduler.step()
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')


    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        # DDP unwrap
        model_for_state = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        # FP32(QAT-준비) state_dict 저장
        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}

        ckpt_fp32 = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': model_for_state.state_dict(),   # QAT-준비 FP32 state_dict
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'version': __version__,
        }
        torch.save(ckpt_fp32, self.wdir / 'last_fp32.pt')

        # 학습 모델 deepcopy → CPU/eval → observer/BN 동결 → convert
        q = deepcopy(model_for_state).cpu()

        # observer 강제 활성화
        q.apply(lambda m: m.enable_observer() if hasattr(m, "enable_observer") else None)

        # dummy forward to convert 전 observer 스케일 보장용
        q.train()       # observer 동작 모드로 전환
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            q(dummy)
        q.eval()        # observer 비활성화 - 추론 모드 전환
        q.apply(disable_observer)
        q.apply(freeze_bn_stats)
        q_int8 = convert(q, inplace=False).eval()     # 여기서 nn.quantized.* 구조로 변환, 양자화된 INT8 모델을 완전히 추론 모드(eval)로 고정

        # INT8 모듈 자체 저장
        torch.save(q_int8, self.wdir / 'last_int8_model.pt')

        torch.cuda.empty_cache()
        gc.collect()

        # best 갱신 시 함께 저장
        if self.best_fitness == self.fitness:
            torch.save(ckpt_fp32, self.wdir / 'best_fp32.pt')
            torch.save(q_int8, self.wdir / 'best_int8.pt')

        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(q_int8, self.wdir / f'epoch{self.epoch}_int8.pt')
            torch.save(ckpt_fp32, self.wdir / f'epoch{self.epoch}_fp32.pt')


    # def final_eval(self):
    #     """Performs final evaluation and validation for object detection YOLO model."""
    #     if dist.is_initialized() and dist.get_rank() != 0:
    #         return    # rank0만 최종 validation 수행 (나머지 rank는 바로 종료)

    #     best_ = self.wdir / 'best_int8.pt'
    #     model = load_quantized_model(best_, self.model_cfg, self.data['nc'])

    #     # DDP 감싸진 모델이면 unwrap
    #     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #         model = model.module

    #     # quantized model은 CPU에서만 실행되어야 함
    #     if hasattr(self.validator, "device"):
    #         self.validator.device = torch.device("cpu")
    #     model.eval()

    #     for p in model.parameters():
    #         p.requires_grad_(False)

    #     # strip optimizers
    #     LOGGER.info(f'\nValidating {best_}...')
    #     self.validator.qat = self.qat     # 추가한 코드
    #     self.validator.args.plots = self.args.plots
    #     self.validator.args.verbose = True  # 클래스별 metric 출력 설정
    #     # validator가 GPU로 올리지 않도록 강제
    #     if hasattr(self.validator, "device"):     # 추가한 코드
    #         self.validator.device = torch.device("cpu")
    #     self.metrics = self.validator(trainer=self, model=model)
    #     self.metrics.pop('fitness', None)
    #     self.run_callbacks('on_fit_epoch_end')
            
    def final_eval(self):
        """
        Performs final evaluation and validation for object detection YOLO model
        after QAT training.

        This version handles quantized ConvReLU2d modules safely and prevents
        AttributeError caused by missing '_modules' in quantized layers.
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return  # rank0만 수행

        print("\n🚀 Starting final evaluation (Safe Mode)...")

        # --- Load the quantized model safely ---
        best_path = self.wdir / 'best_int8.pt'
        if not best_path.exists():
            LOGGER.warning(f"⚠️ {best_path} not found. Skipping final evaluation.")
            return

        # Quantized model은 이미 eval 상태로 저장되어 있으므로 추가 eval 불필요
        model = torch.load(best_path, map_location='cpu')

        # --- Safe wrapper: prevent eval/train recursion errors ---
        class SafeQATWrappedModel(nn.Module):
            def __init__(self, model, names, stride):
                super().__init__()
                self.model = model
                self.names = names
                self.stride = stride if isinstance(stride, torch.Tensor) else torch.tensor([stride])
                self.nc = len(names)
                self.pt = False
                self.amp = False
                self.end2end = False

            def forward(self, x, *args, **kwargs):
                return self.model(x)

            def eval(self):
                try:
                    super().eval()
                except AttributeError:
                    print("⚠️ Skipping eval() on fused quantized modules (ConvReLU2d).")
                return self

            def train(self, mode=True):
                # Quantized 모델은 학습 모드로 전환하지 않음
                print("⚠️ Ignoring train() call for quantized model.")
                return self

            def loss(self, batch, preds):
                return torch.tensor(0.0), torch.zeros(3, device=preds.device)

            def fuse(self, verbose=False):
                return self

        # --- Wrap model safely ---
        wrapped = SafeQATWrappedModel(
            model,
            getattr(model, 'names', {0: 'object'}),
            stride=getattr(model, 'stride', torch.tensor([32]))
        )

        # --- Force CPU-only validation ---
        if hasattr(self.validator, "device"):
            self.validator.device = torch.device("cpu")

        for p in wrapped.parameters():
            p.requires_grad_(False)

        torch.cuda.empty_cache()
        gc.collect()

        # --- Run validation safely ---
        LOGGER.info(f"\n🧩 Validating quantized model: {best_path}")
        self.validator.qat = self.qat
        self.validator.args.plots = self.args.plots
        self.validator.args.verbose = True

        try:
            self.metrics = self.validator(trainer=self, model=wrapped)
            self.metrics.pop('fitness', None)
            LOGGER.info("✅ Final evaluation completed successfully.")
        except Exception as e:
            LOGGER.error(f"❌ Validation failed during final_eval(): {e}")
        finally:
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()
            gc.collect()


# 전역 변수: 로그 기록용
qat_scale_log = defaultdict(list)

def log_qat_observer_status(model: torch.nn.Module, step: int = None):
    """
    QAT 중 FakeQuantObserver들의 scale 변화를 기록하고,
    학습 종료 후 그래프로 저장하는 디버그 유틸리티.
    """
    global qat_scale_log

    step = step or 0
    count = 0
    enabled_count = 0
    disabled_count = 0

    KEY_LAYERS = [
        "model.0.conv.weight_fake_quant",
        "model.0.conv.activation_post_process",
        "model.4.cv1.conv.weight_fake_quant",
        "model.4.cv1.conv.activation_post_process",
        "model.22.dfl.conv.weight_fake_quant",
        "model.22.dfl.conv.activation_post_process",
    ]

    key_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.quantization.FakeQuantize):
            count += 1
            scale = getattr(module, "scale", None)
            if scale is not None and (name.endswith("fake_quant") or name.endswith("activation_post_process")):
                enabled_count += 1
                if any(k in name for k in KEY_LAYERS):
                    key_scales[name] = float(scale.mean())
            else:
                disabled_count += 1

    print(f"\n[QAT DEBUG] Step {step}")
    print(f" - Found {count} FakeQuant modules | Enabled: {enabled_count}, Disabled: {disabled_count}")
    if len(key_scales) == 0:
        print("⚠️ No matching FakeQuant modules with valid scale found.")
    else:
        print(" - Key scales:")
        for k, v in key_scales.items():
            print(f"   • {k:45s} scale ≈ {v:.6f}")
            qat_scale_log[k].append((step, v))
