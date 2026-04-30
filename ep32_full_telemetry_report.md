# SGLATrack Ep32 Full Telemetry Report

Generated: 2026-04-30 17:13:23 +03:00

## Checkpoint

- Path: output_local\checkpoints\train\sglatrack\custom_train_highres_scratch_b172\sglatrack_ep0032.pth.tar
- Size bytes: 98384878
- Last write: 04/30/2026 15:54:49
- SHA256: 473B8B6FA3EC78F2A08EFDA390012929678B9300FF90AB276360EA1B241B14CD

## Validation Summary

- Command: python tools\validate_checkpoint.py --config custom_train_highres_scratch_b172 --checkpoint output_local\checkpoints\train\sglatrack\custom_train_highres_scratch_b172\sglatrack_ep0032.pth.tar --batch_size 32 --val_samples 3000 --num_workers 4 --use_lmdb 0
- Effective samples: 2976
- Dataset mode: folder, split=val

```text
checkpoint: output_local\checkpoints\train\sglatrack\custom_train_highres_scratch_b172\sglatrack_ep0032.pth.tar
epoch: 32
val_samples: 2976
IoU: 0.67459
Loss/giou: 0.37870
Loss/l1: 0.04606
Loss/location: 0.75869
Loss/total: 1.74641
pro_loss: 0.00000
```

## Epoch 32 Train Telemetry From Log

```text
[train: 32, 20 / 87] FPS: 45.2 (54.3)  ,  DataTime: 3.074 (0.000)  ,  ForwardTime: 0.728  ,  TotalTime: 3.802  ,  Loss/total: 0.79188  ,  Loss/giou: 0.19340  ,  Loss/l1: 0.01635  ,  Loss/location: 0.31253  ,  pro_loss: 0.05409  ,  IoU: 0.81731  ,  
[train: 32, 40 / 87] FPS: 49.9 (191.3)  ,  DataTime: 2.727 (0.000)  ,  ForwardTime: 0.717  ,  TotalTime: 3.444  ,  Loss/total: 0.79586  ,  Loss/giou: 0.19348  ,  Loss/l1: 0.01590  ,  Loss/location: 0.31873  ,  pro_loss: 0.05336  ,  IoU: 0.81670  ,  
[train: 32, 60 / 87] FPS: 42.6 (205.3)  ,  DataTime: 3.318 (0.000)  ,  ForwardTime: 0.720  ,  TotalTime: 4.039  ,  Loss/total: 0.79732  ,  Loss/giou: 0.19318  ,  Loss/l1: 0.01603  ,  Loss/location: 0.32018  ,  pro_loss: 0.05313  ,  IoU: 0.81724  ,  
[train: 32, 80 / 87] FPS: 42.0 (203.8)  ,  DataTime: 3.377 (0.000)  ,  ForwardTime: 0.722  ,  TotalTime: 4.100  ,  Loss/total: 0.79944  ,  Loss/giou: 0.19349  ,  Loss/l1: 0.01613  ,  Loss/location: 0.32092  ,  pro_loss: 0.05434  ,  IoU: 0.81706  ,  
[train: 32, 87 / 87] FPS: 40.9 (202.6)  ,  DataTime: 3.482 (0.000)  ,  ForwardTime: 0.721  ,  TotalTime: 4.203  ,  Loss/total: 0.79541  ,  Loss/giou: 0.19225  ,  Loss/l1: 0.01594  ,  Loss/location: 0.32037  ,  pro_loss: 0.05431  ,  IoU: 0.81798  ,  
```

## Config Snapshot

```yaml
DATA:
  MAX_SAMPLE_INTERVAL: 200
  MEAN:
  - 0.485
  - 0.456
  - 0.406
  SEARCH:
    CENTER_JITTER: 3
    FACTOR: 4.0
    NUMBER: 1
    SCALE_JITTER: 0.25
    SIZE: 256
  STD:
  - 0.229
  - 0.224
  - 0.225
  TEMPLATE:
    CENTER_JITTER: 0
    FACTOR: 2.0
    SCALE_JITTER: 0
    SIZE: 128
  TRAIN:
    DATASETS_NAME:
    - CUSTOM
    DATASETS_RATIO:
    - 1
    SAMPLE_PER_EPOCH: 15000
  VAL:
    DATASETS_NAME:
    - CUSTOM
    DATASETS_RATIO:
    - 1
    SAMPLE_PER_EPOCH: 3000
MODEL:
  BACKBONE:
    STRIDE: 16
    TYPE: deit_tiny_distilled_patch16
  EXTRA_MERGER: false
  HEAD:
    NUM_CHANNELS: 256
    TYPE: CENTER
  PRETRAIN_FILE: deit_tiny_distilled_patch16_224-b40b3cf7.pth
  RETURN_INTER: false
TEST:
  EPOCH: 297
  SEARCH_FACTOR: 4.0
  SEARCH_SIZE: 256
  TEMPLATE_FACTOR: 2.0
  TEMPLATE_SIZE: 128
TRAIN:
  AMP: false
  BACKBONE_MULTIPLIER: 0.1
  BATCH_SIZE: 172
  DROP_PATH_RATE: 0.1
  EPOCH: 50
  GIOU_WEIGHT: 2.0
  GRAD_ACCUM_STEPS: 1
  GRAD_CLIP_NORM: 0.1
  L1_WEIGHT: 5.0
  LR: 0.0004
  LR_DROP_EPOCH: 80
  NON_BLOCKING_TRANSFER: true
  NUM_WORKER: 8
  OPTIMIZER: ADAMW
  PREFETCH_FACTOR: 4
  PRINT_INTERVAL: 20
  SCHEDULER:
    DECAY_RATE: 0.1
    TYPE: cosine
  VAL_EPOCH_INTERVAL: 999
  WEIGHT_DECAY: 0.0001
```

## Raw Validation Output

Raw file copy: output_local\reports\ep32_val_3000_raw.txt

```text
python : C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\helpers.py:7: Futu
reWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
At line:2 char:1
+ python tools\validate_checkpoint.py --config custom_train_highres_scr ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (C:\Users\Hezarf...via timm.models:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\layers\__init__.py:49: Futu
reWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\registry.py:4: FutureWarnin
g: Importing from timm.models.registry is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\helpers.py:7: FutureWarning
: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\layers\__init__.py:49: Futu
reWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\registry.py:4: FutureWarnin
g: Importing from timm.models.registry is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\helpers.py:7: FutureWarning
: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\layers\__init__.py:49: Futu
reWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\registry.py:4: FutureWarnin
g: Importing from timm.models.registry is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\helpers.py:7: FutureWarning
: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\layers\__init__.py:49: Futu
reWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\registry.py:4: FutureWarnin
g: Importing from timm.models.registry is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\helpers.py:7: FutureWarning
: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\layers\__init__.py:49: Futu
reWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
C:\Users\HezarfenUAlab\AppData\Local\Programs\Python\Python39\lib\site-packages\timm\models\registry.py:4: FutureWarnin
g: Importing from timm.models.registry is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
D:\visiontrak\SGLATrack\lib\train\data\loader.py:85: UserWarning: TypedStorage is deprecated. It will be removed in the
 future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages dir
ectly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  storage = batch[0].storage()._new_shared(numel)
D:\visiontrak\SGLATrack\lib\train\data\loader.py:85: UserWarning: TypedStorage is deprecated. It will be removed in the
 future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages dir
ectly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  storage = batch[0].storage()._new_shared(numel)
D:\visiontrak\SGLATrack\lib\train\data\loader.py:85: UserWarning: TypedStorage is deprecated. It will be removed in the
 future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages dir
ectly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  storage = batch[0].storage()._new_shared(numel)
D:\visiontrak\SGLATrack\lib\train\data\loader.py:85: UserWarning: TypedStorage is deprecated. It will be removed in the
 future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages dir
ectly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  storage = batch[0].storage()._new_shared(numel)
sampler_mode causal
Building Custom dataset (split=train)
CustomDataset [train]: 253 sekans yuklendi (zorluk sirasiyla)
[WARN] CustomDataset: frame/annotation sayisi farkli, dataset3_group4_1 ilk 298 oge kullanilacak
Building Custom dataset (split=val)
CustomDataset [val]: 44 sekans yuklendi (zorluk sirasiyla)
[WARN] CustomDataset: frame/annotation sayisi farkli, dataset2_RcCar6 ilk 210 oge kullanilacak
[WARN] CustomDataset: frame/annotation sayisi farkli, dataset2_Surfing03 ilk 117 oge kullanilacak
[WARN] CustomDataset: frame/annotation sayisi farkli, dataset3_car16_3 ilk 168 oge kullanilacak
[WARN] CustomDataset: frame/annotation sayisi farkli, dataset3_car4 ilk 1302 oge kullanilacak
[] ['dist_token', 'head_dist.weight', 'head_dist.bias']
Load pretrained model from: D:\visiontrak\SGLATrack\lib\models\sglatrack\../../../pretrained_models\deit_tiny_distilled_patch16_224-b40b3cf7.pth
Timm patch embedding is reload!
[val] step 5/93 samples=160
[val] step 10/93 samples=320
[val] step 15/93 samples=480
[val] step 20/93 samples=640
[val] step 25/93 samples=800
[val] step 30/93 samples=960
[val] step 35/93 samples=1120
[val] step 40/93 samples=1280
[val] step 45/93 samples=1440
[val] step 50/93 samples=1600
[val] step 55/93 samples=1760
[val] step 60/93 samples=1920
[val] step 65/93 samples=2080
[val] step 70/93 samples=2240
[val] step 75/93 samples=2400
[val] step 80/93 samples=2560
[val] step 85/93 samples=2720
[val] step 90/93 samples=2880
[val] step 93/93 samples=2976
checkpoint: output_local\checkpoints\train\sglatrack\custom_train_highres_scratch_b172\sglatrack_ep0032.pth.tar
epoch: 32
val_samples: 2976
IoU: 0.67459
Loss/giou: 0.37870
Loss/l1: 0.04606
Loss/location: 0.75869
Loss/total: 1.74641
pro_loss: 0.00000
```
## Checkpoint Internal Metadata

`	ext
keys: ['actor_type', 'constructor', 'epoch', 'net', 'net_info', 'net_type', 'optimizer', 'settings', 'stats']
epoch: 32
actor_type: sglatrackActor
net_type: sglatrack
optimizer_keys: ['param_groups', 'state']
optimizer_state_count: 208
stats_type: OrderedDict
stats_repr: OrderedDict([('train', OrderedDict([('Loss/total', <lib.train.admin.stats.AverageMeter object at 0x000002DC18265340>), ('Loss/giou', <lib.train.admin.stats.AverageMeter object at 0x000002DC182653A0>), ('Loss/l1', <lib.train.admin.stats.AverageMeter object at 0x000002DC182653D0>), ('Loss/location', <lib.train.admin.stats.AverageMeter object at 0x000002DC18265430>), ('pro_loss', <lib.train.admin.stats.AverageMeter object at 0x000002DC18265400>), ('IoU', <lib.train.admin.stats.AverageMeter object at 0x000002DC182654C0>), ('LearningRate/group0', <lib.train.admin.stats.StatValue object at 0x000002DC18265370>), ('LearningRate/group1', <lib.train.admin.stats.StatValue object at 0x000002DC18265520>)])), ('val', None)])
`
