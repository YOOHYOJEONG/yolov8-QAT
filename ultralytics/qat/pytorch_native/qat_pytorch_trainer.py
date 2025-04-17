from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, get_default_qat_qconfig, convert, disable_observer, fuse_modules_qat
from torch.nn.intrinsic.qat import freeze_bn_stats

import time
import math
import copy
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

from ultralytics.qat.pytorch_native.quant_pytorch_ops import quant_module_change, BACKEND, WORLD_SIZE


# QAT 후 convert()된 모델을 validator와 호환되도록 감싸는 래퍼 클래스
class QATWrappedModel(nn.Module):
    def __init__(self, quant_model, names, stride):
        super().__init__()
        self.model = quant_model
        self.names = names
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

def convert2qat(model):
    _model = deepcopy(model).to("cpu").eval()
    #_model.apply(disable_observer)
    quantized_model = convert(_model, inplace=False)
    quantized_model.eval()

    return quantized_model

def load_quantized_model(path, cfg, nc):

    Conv.default_act = nn.ReLU()
    dicts = torch.load(path, map_location='cpu')

    model = QuantYolo(cfg = cfg, ch = 3, nc = nc, verbose = False)
    model.fuse_model()
    model.qconfig = get_default_qat_qconfig(dicts['backend'])
    prepare_qat(model, inplace =True)

    # Load weights before convert
    model.load_state_dict(dicts['model'], strict=False)
    quantized_model = convert(model, inplace=False)
    quantized_model.eval()

    print("✅ Quantized model successfully loaded and converted.")
    wrapped_model = QATWrappedModel(quantized_model, model.names, stride=torch.tensor([32]))
    
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
        """_summary_

        Args:
            cfg (_type_, optional): Student model configuration. Defaults to None.
            weights (_type_, optional): Student weight configuration. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        Conv.default_act = nn.ReLU()
        self.model_cfg = cfg
        model = QuantYolo(cfg = cfg, ch = 3, nc = self.data['nc'], verbose = False)
        if weights:
            print(f"➰ Loading weights from: {weights}")
            pretrained = attempt_load_weights(weights, device="cpu", inplace=False)
            pretrained_dict = pretrained.state_dict()
            model_dict = model.state_dict()

            # shape 맞는 것만 골라서 로드
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            skipped = [k for k in pretrained_dict if k not in filtered_dict]
            if skipped:
                print(f"⚠️ Skipped loading {len(skipped)} mismatched layers: {skipped[:3]}...")

            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)

        # quant_module_change(model)
        model.fuse_model()
        model.qconfig = get_default_qat_qconfig(backend)
        model.backend = backend
        prepare_qat(model, inplace = True)

        # QAT 후 변환 테스트 (이건 실제 학습 모델 아님, 그냥 변환 테스트)
        quantized_model = convert(model.to("cpu").eval(), inplace=False)
        quantized_model.eval()
        quantized_model(torch.randn(1,3,640,640, dtype=torch.float))

        print(f"Using QConfig for {backend} backend")

        return model
    
    def setup_model(self):

        return None

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= self.epochs * 3.0, eta_min=self.args.lr0 * 0.01)

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        self.args.warmup_epochs = 0
        self.epochs = self.epochs // 10
        self.args.lr0 /= 50
        self.args.optimizer = 'SGD'

        # Model
        self.run_callbacks('on_pretrain_routine_start')

        ckpt = self.setup_model()
        self.model = self.model.to(self.device)

        self.set_model_attributes()

        # Freeze layers
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
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
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
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
            if epoch > 3:
                # Freeze batch norm mean and variance estimates
                self.model.apply(freeze_bn_stats)
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
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
                with torch.cuda.amp.autocast(self.amp):
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
        import pandas as pd  # scope for faster startup
        
        # 1. 순수 QuantYolo 모델 새로 생성 (DDP와 무관)
        pure_model = QuantYolo(cfg=self.model_cfg, ch=3, nc=self.data['nc'], verbose=False)
        pure_model.eval()
        pure_model.fuse_model()
        pure_model.qconfig = get_default_qat_qconfig(getattr(self.model, 'backend', 'qnnpack'))
        pure_model.backend = getattr(self.model, 'backend', 'qnnpack')

        # 2. DDP 여부와 상관없이 state_dict만 복사
        model_for_state = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        pure_model.load_state_dict(model_for_state.state_dict(), strict=False)

        # 3. convert()만 수행, forward()는 하지 않음
        quantized_model = convert(pure_model.cpu(), inplace=False)
        quantized_model.eval()

        # 4. 원래 모델은 다시 GPU로 복원
        self.model.to(self.device)

        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
        
        ckpt_int8 = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': quantized_model.state_dict(),   # qat int8 model save
            'backend': model_for_state.backend,
            # 'ema': self.ema.ema.state_dict(),
            # 'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'version': __version__}
        
        ckpt_fp32 = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': model_for_state.state_dict(),   # qat fp32 model save
            'backend': model_for_state.backend,
            # 'ema': self.ema.ema.state_dict(),
            # 'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last and best
        torch.save(ckpt_int8, self.wdir / 'last_int8.pt')
        torch.save(ckpt_fp32, self.wdir / 'last_fp32.pt')
        if self.best_fitness == self.fitness:
            torch.save(ckpt_fp32, self.wdir / 'best_fp32.pt')
            torch.save(ckpt_int8, self.wdir / 'best_int8.pt')
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt_int8, self.wdir / f'epoch{self.epoch}_ing8.pt')
            torch.save(ckpt_fp32, self.wdir / f'epoch{self.epoch}_fp32.pt')

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        best_ = self.wdir / 'best_int8.pt'
        model = load_quantized_model(best_, self.model_cfg, self.data['nc'])

        # DDP 감싸진 모델이면 unwrap
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        # quantized model은 CPU에서만 실행되어야 함
        model.to("cpu")
        model.eval()

        # strip optimizers
        LOGGER.info(f'\nValidating {best_}...')
        self.validator.qat = self.qat     # 추가한 코드
        self.validator.args.plots = self.args.plots
        self.validator.args.verbose = True  # 클래스별 metric 출력 설정
        # validator가 GPU로 올리지 않도록 강제
        if hasattr(self.validator, "device"):     # 추가한 코드
            self.validator.device = torch.device("cpu")
        self.metrics = self.validator(trainer=self, model=model)
        self.metrics.pop('fitness', None)
        self.run_callbacks('on_fit_epoch_end')
