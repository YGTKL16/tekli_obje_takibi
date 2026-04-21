# SGLATrack Training Pipeline

[< Back to Index](README.md)

## Datasets

### Training Datasets

| Dataset | Ratio | Description |
|---------|-------|-------------|
| LaSOT | 1 | Large-scale single object tracking (1,400 sequences) |
| GOT-10k (vottrain) | 1 | Generic object tracking (10,000 sequences) |
| COCO17 | 1 | Static image pairs from COCO 2017 detection |
| TrackingNet | 1 | Large-scale short-term tracking (30,000 sequences) |

**Samples per epoch**: 60,000
**Sampling mode**: `causal` — sequential frame pairs with max interval of 200 frames

### Data Directory Structure

```
models/SGLATrack/data/
├── lasot/
│   ├── airplane/
│   ├── basketball/
│   ├── bear/
│   └── ...
├── got10k/
│   ├── test/
│   ├── train/
│   └── val/
├── coco/
│   ├── annotations/
│   └── images/
└── trackingnet/
    ├── TRAIN_0
    ├── TRAIN_1
    ├── ...
    ├── TRAIN_11
    └── TEST
```

### Validation

| Dataset | Ratio | Samples/epoch |
|---------|-------|---------------|
| GOT-10k (votval) | 1 | 10,000 |

## Configuration System

SGLATrack uses a two-tier configuration:

1. **Default config** (`lib/config/sglatrack/config.py`) — all parameters with defaults
2. **Experiment YAML** (`experiments/sglatrack/deit_distilled.yaml`) — overrides for specific variants

### Default Parameters (ViT-Base)

| Category | Parameter | Default |
|----------|-----------|---------|
| **Model** | `PRETRAIN_FILE` | `mae_pretrain_vit_base.pth` |
| **Backbone** | `TYPE` | `vit_base_patch16_224` |
| | `STRIDE` | 16 |
| | `CAT_MODE` | `direct` |
| **Head** | `TYPE` | `CENTER` |
| | `NUM_CHANNELS` | 256 |
| **Data** | `SEARCH.SIZE` | 320 |
| | `SEARCH.FACTOR` | 5.0 |
| | `TEMPLATE.SIZE` | 128 |
| | `TEMPLATE.FACTOR` | 2.0 |
| **Training** | `EPOCH` | 500 |
| | `BATCH_SIZE` | 16 |
| | `LR` | 0.0001 |
| | `LR_DROP_EPOCH` | 400 |
| | `WEIGHT_DECAY` | 0.0001 |
| | `BACKBONE_MULTIPLIER` | 0.1 |
| | `DROP_PATH_RATE` | 0.1 |
| | `GRAD_CLIP_NORM` | 0.1 |
| | `OPTIMIZER` | `ADAMW` |

### DeiT-Tiny Distilled Overrides

Key differences from defaults (`experiments/sglatrack/deit_distilled.yaml`):

| Parameter | Default | DeiT-Tiny |
|-----------|---------|-----------|
| `PRETRAIN_FILE` | `mae_pretrain_vit_base.pth` | `deit_tiny_distilled_patch16_224.pth` |
| `BACKBONE.TYPE` | `vit_base_patch16_224` | `deit_tiny_distilled_patch16` |
| `SEARCH.SIZE` | 320 | **256** |
| `SEARCH.FACTOR` | 5.0 | **4.0** |
| `SEARCH.CENTER_JITTER` | 4.5 | **3** |
| `SEARCH.SCALE_JITTER` | 0.5 | **0.25** |
| `BATCH_SIZE` | 16 | **32** |
| `EPOCH` | 500 | **300** |
| `LR` | 0.0001 | **0.0004** |
| `LR_DROP_EPOCH` | 400 | **240** |
| `TEST.EPOCH` | 500 | **297** |
| `TEST.SEARCH_FACTOR` | 5.0 | **4.0** |
| `TEST.SEARCH_SIZE` | 320 | **256** |

## Training Pipeline

### Entry Point

```bash
python tracking/train.py \
  --script sglatrack \
  --config deit_distilled \
  --save_dir ./output \
  --mode multiple --nproc_per_node 4 \
  --use_wandb 0
```

**Arguments**:
- `--script`: Training script name (maps to `lib/train/train_script.py`)
- `--config`: Experiment config under `experiments/sglatrack/`
- `--mode`: `single` (1 GPU) or `multiple` (DDP multi-GPU)
- `--nproc_per_node`: Number of GPUs for distributed training
- `--use_wandb`: Enable Weights & Biases logging (0=off, 1=on)

### Script Hierarchy

```
tracking/train.py                   # CLI entry point, DDP launch
  └── lib/train/run_training.py     # Distributed coordinator
        └── lib/train/train_script.py  # Dataset setup, model build, training loop
              └── lib/train/trainers/ltr_trainer.py  # Base training loop
```

### Loss Computation

The `sglatrackActor` class (`lib/train/actors/sglatrack.py`) handles the forward pass and loss computation:

```
Input:
  template_images: (1, batch, 3, 128, 128)
  search_images:   (1, batch, 3, 256, 256)
  search_anno:     (1, batch, 4) — [x, y, w, h] ground truth

Forward:
  → Network produces: pred_boxes, score_map, size_map, offset_map, cos_tensor, pro

Loss:
  gt_gaussian_maps = generate_heatmap(search_anno)

  L_giou = giou_loss(pred_boxes, gt_boxes)           × 2.0
  L_l1   = l1_loss(pred_boxes, gt_boxes)              × 5.0
  L_focal = focal_loss(score_map, gt_gaussian_maps)   × 1.0
  L_cos  = l1_loss(pro, pro_target)                   × 0.2

  L_total = L_giou + L_l1 + L_focal + L_cos
```

### Optimizer and Scheduler

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW |
| Learning rate | 0.0004 (DeiT) |
| Backbone LR | 0.0004 × 0.1 = 0.00004 |
| Weight decay | 0.0001 |
| Gradient clipping | norm = 0.1 |
| Scheduler | StepLR |
| LR decay rate | 0.1 at epoch 240 |
| AMP | Disabled |

### Pre-trained Weights

```bash
# DeiT-Tiny Distilled (Meta/Facebook Research)
wget https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth \
  -P models/SGLATrack/pretrained_models/
```

The model loads pre-trained weights during `build_sglatrack()`. Backbone weights are loaded first, then `finetune_track()` adapts patch embedding and position embeddings to the tracking input sizes.

### Logging

| Platform | Flag | Description |
|----------|------|-------------|
| WandB | `--use_wandb 1` | Detailed experiment tracking, loss curves, hyperparameters |
| TensorBoard | Built-in | Training loss, validation metrics via `torch.utils.tensorboard` |
| Console | `PRINT_INTERVAL=50` | Print loss every 50 iterations |

### Path Configuration

After running `create_default_local_file.py`, modify paths in:
- `lib/train/admin/local.py` — training data paths, save directory
- `lib/test/evaluation/local.py` — test data paths, results directory

---

# SGLATrack Eğitim Pipeline'i

[< İndeks'e Dön](README.md)

## Veri Setleri

### Eğitim Veri Setleri

| Veri Seti | Oran | Açıklama |
|-----------|------|----------|
| LaSOT | 1 | Büyük ölçekli tek nesne takibi (1.400 sekans) |
| GOT-10k (vottrain) | 1 | Genel nesne takibi (10.000 sekans) |
| COCO17 | 1 | COCO 2017 tespitinden statik görüntü ciftleri |
| TrackingNet | 1 | Büyük ölçekli kisa vadeli takip (30.000 sekans) |

**Epoch başına ornek**: 60.000
**Ornekleme modu**: `causal` — maksimum 200 kare aralikli ardisik kare ciftleri

### Veri Dizin Yapisi

```
models/SGLATrack/data/
├── lasot/
│   ├── airplane/
│   ├── basketball/
│   ├── bear/
│   └── ...
├── got10k/
│   ├── test/
│   ├── train/
│   └── val/
├── coco/
│   ├── annotations/
│   └── images/
└── trackingnet/
    ├── TRAIN_0
    ├── TRAIN_1
    ├── ...
    ├── TRAIN_11
    └── TEST
```

### Dogrulama

| Veri Seti | Oran | Ornek/epoch |
|-----------|------|-------------|
| GOT-10k (votval) | 1 | 10.000 |

## Konfigürasyon Sistemi

SGLATrack iki katmanli konfigürasyon kullanır:

1. **Varsayılan konfig** (`lib/config/sglatrack/config.py`) — varsayılan değerli tüm parametreler
2. **Deney YAML** (`experiments/sglatrack/deit_distilled.yaml`) — belirli varyantlar için geçersiz kilmalar

### Varsayılan Parametreler (ViT-Base)

| Kategori | Parametre | Varsayılan |
|----------|-----------|------------|
| **Model** | `PRETRAIN_FILE` | `mae_pretrain_vit_base.pth` |
| **Omurga** | `TYPE` | `vit_base_patch16_224` |
| | `STRIDE` | 16 |
| | `CAT_MODE` | `direct` |
| **Başlık** | `TYPE` | `CENTER` |
| | `NUM_CHANNELS` | 256 |
| **Veri** | `SEARCH.SIZE` | 320 |
| | `SEARCH.FACTOR` | 5.0 |
| | `TEMPLATE.SIZE` | 128 |
| | `TEMPLATE.FACTOR` | 2.0 |
| **Eğitim** | `EPOCH` | 500 |
| | `BATCH_SIZE` | 16 |
| | `LR` | 0.0001 |
| | `LR_DROP_EPOCH` | 400 |
| | `WEIGHT_DECAY` | 0.0001 |
| | `BACKBONE_MULTIPLIER` | 0.1 |
| | `DROP_PATH_RATE` | 0.1 |
| | `GRAD_CLIP_NORM` | 0.1 |
| | `OPTIMIZER` | `ADAMW` |

### DeiT-Tiny Distilled Gecersiz Kilmalar

Varsayılanlardan temel farklar (`experiments/sglatrack/deit_distilled.yaml`):

| Parametre | Varsayılan | DeiT-Tiny |
|-----------|------------|-----------|
| `PRETRAIN_FILE` | `mae_pretrain_vit_base.pth` | `deit_tiny_distilled_patch16_224.pth` |
| `BACKBONE.TYPE` | `vit_base_patch16_224` | `deit_tiny_distilled_patch16` |
| `SEARCH.SIZE` | 320 | **256** |
| `SEARCH.FACTOR` | 5.0 | **4.0** |
| `SEARCH.CENTER_JITTER` | 4.5 | **3** |
| `SEARCH.SCALE_JITTER` | 0.5 | **0.25** |
| `BATCH_SIZE` | 16 | **32** |
| `EPOCH` | 500 | **300** |
| `LR` | 0.0001 | **0.0004** |
| `LR_DROP_EPOCH` | 400 | **240** |
| `TEST.EPOCH` | 500 | **297** |
| `TEST.SEARCH_FACTOR` | 5.0 | **4.0** |
| `TEST.SEARCH_SIZE` | 320 | **256** |

## Eğitim Pipeline'i

### Giriş Noktasi

```bash
python tracking/train.py \
  --script sglatrack \
  --config deit_distilled \
  --save_dir ./output \
  --mode multiple --nproc_per_node 4 \
  --use_wandb 0
```

**Argümanlar**:
- `--script`: Eğitim betik adi (`lib/train/train_script.py`'a esler)
- `--config`: `experiments/sglatrack/` altindaki deney konfigürasyonu
- `--mode`: `single` (1 GPU) veya `multiple` (DDP coklu-GPU)
- `--nproc_per_node`: Dagitik eğitim için GPU sayısı
- `--use_wandb`: Weights & Biases günlüklemeyi etkinlestir (0=kapali, 1=acik)

### Betik Hiyerarsisi

```
tracking/train.py                   # CLI giriş noktasi, DDP baslatma
  └── lib/train/run_training.py     # Dagitik koordinator
        └── lib/train/train_script.py  # Veri seti kurulumu, model inşasi, eğitim dongusu
              └── lib/train/trainers/ltr_trainer.py  # Temel eğitim dongusu
```

### Kayıp Hesaplama

`sglatrackActor` sınıfı (`lib/train/actors/sglatrack.py`) ileri gecis ve kayıp hesaplamasini yönetir:

```
Giriş:
  şablon_görüntüler:  (1, yığıt, 3, 128, 128)
  arama_görüntüler:   (1, yığıt, 3, 256, 256)
  arama_etiket:       (1, yığıt, 4) — [x, y, w, h] gercek değer

İleri Gecis:
  → Ag ciktilar: tahmin_kutular, skor_haritası, boyut_haritası, ofset_haritası, cos_tensor, pro

Kayıp:
  gt_gaussian_maps = generate_heatmap(arama_etiket)

  L_giou  = giou_loss(tahmin_kutular, gercek_kutular)    × 2.0
  L_l1    = l1_loss(tahmin_kutular, gercek_kutular)       × 5.0
  L_focal = focal_loss(skor_haritası, gt_gaussian_maps)   × 1.0
  L_cos   = l1_loss(pro, pro_hedef)                       × 0.2

  L_toplam = L_giou + L_l1 + L_focal + L_cos
```

### Optimizer ve Zamanlayici

| Bilesen | Ayar |
|---------|------|
| Optimizer | AdamW |
| Ogrenme hizi | 0.0004 (DeiT) |
| Omurga ogrenme hizi | 0.0004 × 0.1 = 0.00004 |
| Ağırlık azaltma | 0.0001 |
| Gradyan kirpma | norm = 0.1 |
| Zamanlayici | StepLR |
| Ogrenme hizi düsüs orani | Epoch 240'ta 0.1 |
| AMP | Devre disi |

### Önceden Eğitilmiş Ağırlıklar

```bash
# DeiT-Tiny Distilled (Meta/Facebook Research)
wget https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth \
  -P models/SGLATrack/pretrained_models/
```

Model, `build_sglatrack()` sırasında onceden eğitilmiş ağırlıkları yükler. Omurga ağırlıkları önce yuklenir, ardından `finetune_track()` yama gömme ve konum gömmelerini takip giriş boyutlarına uyarlar.

### Günlükleme

| Platform | Bayrak | Açıklama |
|----------|--------|----------|
| WandB | `--use_wandb 1` | Detayli deney takibi, kayıp egrileri, hiperparametreler |
| TensorBoard | Dahili | `torch.utils.tensorboard` ile eğitim kaybi, dogrulama metrikleri |
| Konsol | `PRINT_INTERVAL=50` | Her 50 iterasyonda kayıp yazdir |

### Yol Konfigürasyonu

`create_default_local_file.py` calistirildiktan sonra, yollari su dosyalarda degistirin:
- `lib/train/admin/local.py` — eğitim verisi yollari, kayit dizini
- `lib/test/evaluation/local.py` — test verisi yollari, sonüç dizini
