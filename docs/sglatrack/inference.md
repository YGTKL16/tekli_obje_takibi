# SGLATrack Inference & Tracking

[< Back to Index](README.md)

## Tracker Class

The inference tracker is implemented in `models/SGLATrack/lib/test/tracker/sglatrack.py` as the `sglatrack(BaseTracker)` class.

### Initialization

```python
tracker = sglatrack(params, dataset_name)
```

During construction:
1. Builds the SGLATrack network via `build_sglatrack(cfg, training=False)`
2. Loads checkpoint weights: `network.load_state_dict(ckpt['net'], strict=True)`
3. Moves model to CUDA and sets `eval()` mode
4. Creates a `Preprocessor` for image normalization
5. Generates a **Hann window** (`hann2d`) of size `feat_sz x feat_sz` for spatial response smoothing

### Parameters

Loaded from `lib/test/parameter/sglatrack.py`:

| Parameter | DeiT-Tiny Value | Description |
|-----------|----------------|-------------|
| `template_factor` | 2.0 | Context multiplier for template crop |
| `template_size` | 128 | Template input resolution (pixels) |
| `search_factor` | 4.0 | Context multiplier for search region crop |
| `search_size` | 256 | Search region input resolution (pixels) |
| `checkpoint` | `sglatrack_ep0297.pth.tar` | Model weights file |

**Checkpoint path pattern**:
```
output/checkpoints/train/sglatrack/{config_name}/sglatrack_ep{EPOCH:04d}.pth.tar
```

## Frame-by-Frame Tracking Pipeline

### Step 1: Initialize (First Frame)

```python
tracker.initialize(image, {'init_bbox': [x, y, w, h]})
```

1. **Extract template patch**: Crop around target bbox with `template_factor=2.0` context
   ```python
   z_patch, resize_factor, z_amask = sample_target(
       image, init_bbox, template_factor, output_sz=128
   )
   ```
2. **Preprocess**: Normalize with ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Store template features**: `z_dict1 = preprocessor.process(z_patch, z_amask)` — a `NestedTensor` with image tensor and attention mask
4. **CE mask** (if enabled): Transform bbox to crop coordinates and generate candidate elimination mask
5. **Save initial state**: `self.state = init_bbox` in `[x, y, w, h]` top-left format

### Step 2: Track (Subsequent Frames)

```python
result = tracker.track(image)
bbox = result['target_bbox']  # [x, y, w, h] top-left
```

Full processing pipeline per frame:

```
Frame (H, W, 3)
    |
    v
1. sample_target(frame, prev_state, search_factor=4.0, output_sz=256)
    → x_patch (256, 256, 3), resize_factor, x_amask
    |
    v
2. preprocessor.process(x_patch, x_amask)
    → x_dict: NestedTensor (normalized, CUDA)
    |
    v
3. network.forward_test(template=z_dict1.tensors, search=x_dict.tensors)
    → out_dict: {score_map, size_map, offset_map, backbone_feat}
    |
    v
4. response = output_window * score_map   (Hann window smoothing)
    |
    v
5. box_head.cal_bbox(response, size_map, offset_map)
    → pred_boxes (cx, cy, w, h) normalized [0, 1]
    |
    v
6. Scale to search region: pred_box * search_size / resize_factor
    |
    v
7. map_box_back(): Convert from search-region coords to image coords
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    |
    v
8. clip_box(): Clip to image boundaries (margin=10)
    → final_bbox [x, y, w, h] top-left
```

### Coordinate Mapping

The `map_box_back()` method maps predictions from search region coordinates back to the original image:

```python
def map_box_back(self, pred_box, resize_factor):
    # Previous center
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    
    # Map to image coordinates
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    
    return [cx_real - 0.5*w, cy_real - 0.5*h, w, h]
```

## Running Evaluation

### Test Command

```bash
# General format
python tracking/test.py tracker_name tracker_param \
  --dataset_name DATASET --threads N --num_gpus N

# Examples
python tracking/test.py sglatrack deit_distilled --dataset_name uav123 --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uavdt --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name dtb70 --threads 8 --num_gpus 4
```

### Supported Datasets

| Dataset ID | Full Name | Type |
|-----------|-----------|------|
| `uav123` | UAV123 | Aerial |
| `uav123_10fps` | UAV123 at 10 FPS | Aerial |
| `uavdt` | UAVDT | Aerial |
| `uavtrack112` | UAVTrack112 (V4RFlight112) | Aerial |
| `uavtrack` | UAVTrack_L | Aerial |
| `dtb70` | DTB70 | Aerial |
| `visdrone` | VisDrone2018-SOT | Aerial |
| `lasot` | LaSOT | Generic |
| `got10k_test` | GOT-10k Test | Generic |
| `trackingnet` | TrackingNet | Generic |

### Test Data Structure

```
models/SGLATrack/data/
├── UAV123/
│   ├── anno/
│   └── data_seq/
├── UAV123_10fps/
│   ├── anno/
│   └── data_seq/
├── uavdt/
│   ├── anno/
│   └── sequences/
├── V4RFlight112/
│   ├── anno/
│   ├── anno_l/
│   ├── data_seq/
│   └── attributes/
├── DTB70/
│   ├── Animal1/
│   └── ...
└── VisDrone2018-SOT-test-dev/
    ├── annotations/
    ├── sequences/
    └── attributes/
```

Test datasets available at: [Baidu Pan](https://pan.baidu.com/s/1MaeGLRcAUbJxksbF_CrOeQ?pwd=5vbv) (code: 5vbv)

### Analysis

```bash
python tracking/analysis_results.py
# Requires editing tracker configs and names inside the script
```

## Model Profiling

```bash
python tracking/profile_model.py
```

Measures:
- **FLOPs**: Floating point operations per forward pass
- **Speed**: Frames per second on target GPU
- **Note**: Paper results were measured on a single RTX 2080Ti

## Debug Mode

Pass `--debug 1` to enable visualization:

- **Without Visdom** (`debug=1, use_visdom=False`): Saves annotated frames to `debug/` directory
- **With Visdom** (`debug=1, use_visdom=True`): Live visualization of:
  - Tracking result overlay
  - Search region crop
  - Template patch
  - Raw score map heatmap
  - Hann-windowed score map heatmap
  - Candidate elimination masked search (if CE enabled)

---

# SGLATrack Çıkarım ve Takip

[< İndeks'e Dön](README.md)

## Takipci Sinifi

Çıkarım takipcisi `models/SGLATrack/lib/test/tracker/sglatrack.py` dosyasında `sglatrack(BaseTracker)` sınıfı olarak uygulanmıştır.

### Baslatma

```python
tracker = sglatrack(params, dataset_name)
```

Olusturma sırasında:
1. `build_sglatrack(cfg, training=False)` ile SGLATrack agini olusturur
2. Kontrol noktasi ağırlıklarıni yükler: `network.load_state_dict(ckpt['net'], strict=True)`
3. Modeli CUDA'ya tasir ve `eval()` moduna gecer
4. Görüntü normalizasyonu için `Preprocessor` olusturur
5. Mekansal yanit yumusatmasi için `feat_sz x feat_sz` boyutunda **Hann penceresi** (`hann2d`) uretir

### Parametreler

`lib/test/parameter/sglatrack.py` dosyasından yuklenir:

| Parametre | DeiT-Tiny Değeri | Açıklama |
|-----------|-----------------|----------|
| `template_factor` | 2.0 | Şablon kirpma için baglam carpani |
| `template_size` | 128 | Şablon giriş cozunurlugu (piksel) |
| `search_factor` | 4.0 | Arama bölgesi kirpma için baglam carpani |
| `search_size` | 256 | Arama bölgesi giriş cozunurlugu (piksel) |
| `checkpoint` | `sglatrack_ep0297.pth.tar` | Model ağırlıkları dosyasi |

**Kontrol noktasi yol deseni**:
```
output/checkpoints/train/sglatrack/{konfig_adi}/sglatrack_ep{EPOCH:04d}.pth.tar
```

## Kare Kare Takip Pipeline'i

### Adim 1: Baslatma (Ilk Kare)

```python
tracker.initialize(görüntü, {'init_bbox': [x, y, w, h]})
```

1. **Şablon yamasi çıkar**: Hedef sinir kutusunun etrafini `template_factor=2.0` baglamiyla kirp
   ```python
   z_patch, resize_factor, z_amask = sample_target(
       görüntü, başlangıç_kutusu, template_factor, output_sz=128
   )
   ```
2. **On işleme**: ImageNet istatistikleriyle normalize et (ortalama=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Şablon özelliklerini sakla**: `z_dict1 = preprocessor.process(z_patch, z_amask)` — görüntü tensoru ve dikkat maskesi iceren `NestedTensor`
4. **CE maskesi** (etkinse): Sinir kutusunu kirpma koordinatlarina donustur ve aday eleme maskesi olustur
5. **Başlangıç durumunu kaydet**: `self.state = init_bbox`, `[x, y, w, h]` sol-üst format

### Adim 2: Takip (Sonraki Kareler)

```python
sonüç = tracker.track(görüntü)
kutu = sonuc['target_bbox']  # [x, y, w, h] sol-üst
```

Kare başına tam işleme pipeline'i:

```
Kare (H, W, 3)
    |
    v
1. sample_target(kare, onceki_durum, search_factor=4.0, output_sz=256)
    → x_yamasi (256, 256, 3), yeniden_boyutlama_faktoru, x_amask
    |
    v
2. preprocessor.process(x_yamasi, x_amask)
    → x_dict: NestedTensor (normalize, CUDA)
    |
    v
3. network.forward_test(template=z_dict1.tensors, search=x_dict.tensors)
    → out_dict: {skor_haritası, boyut_haritası, ofset_haritası, omurga_ozellik}
    |
    v
4. yanit = çıkış_penceresi * skor_haritası   (Hann pencere yumusatmasi)
    |
    v
5. box_head.cal_bbox(yanit, boyut_haritası, ofset_haritası)
    → tahmin_kutular (cx, cy, w, h) normalize [0, 1]
    |
    v
6. Arama bölgesine ölçekle: tahmin_kutu * arama_boyutu / yeniden_boyutlama_faktoru
    |
    v
7. map_box_back(): Arama bölgesi koordinatlarindan görüntü koordinatlarina donustur
    cx_gercek = cx + (cx_onceki - yarim_kenar)
    cy_gercek = cy + (cy_onceki - yarim_kenar)
    |
    v
8. clip_box(): Görüntü sinrlarina kirp (marj=10)
    → son_kutu [x, y, w, h] sol-üst
```

### Koordinat Esleme

`map_box_back()` metodu, tahminleri arama bölgesi koordinatlarindan orijinal görüntüye esler:

```python
def map_box_back(self, tahmin_kutu, yeniden_boyutlama_faktoru):
    # Onceki merkez
    cx_onceki = durum[0] + 0.5 * durum[2]
    cy_onceki = durum[1] + 0.5 * durum[3]
    
    cx, cy, w, h = tahmin_kutu
    yarim_kenar = 0.5 * arama_boyutu / yeniden_boyutlama_faktoru
    
    # Görüntü koordinatlarina esle
    cx_gercek = cx + (cx_onceki - yarim_kenar)
    cy_gercek = cy + (cy_onceki - yarim_kenar)
    
    return [cx_gercek - 0.5*w, cy_gercek - 0.5*h, w, h]
```

## Değerlendirme Çalıştırma

### Test Komutu

```bash
# Genel format
python tracking/test.py takipci_adi takipci_param \
  --dataset_name VERI_SETI --threads N --num_gpus N

# Örnekler
python tracking/test.py sglatrack deit_distilled --dataset_name uav123 --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uavdt --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name dtb70 --threads 8 --num_gpus 4
```

### Desteklenen Veri Setleri

| Veri Seti ID | Tam Ad | Tür |
|-------------|--------|-----|
| `uav123` | UAV123 | Hava |
| `uav123_10fps` | UAV123, 10 FPS | Hava |
| `uavdt` | UAVDT | Hava |
| `uavtrack112` | UAVTrack112 (V4RFlight112) | Hava |
| `uavtrack` | UAVTrack_L | Hava |
| `dtb70` | DTB70 | Hava |
| `visdrone` | VisDrone2018-SOT | Hava |
| `lasot` | LaSOT | Genel |
| `got10k_test` | GOT-10k Test | Genel |
| `trackingnet` | TrackingNet | Genel |

### Test Verisi Yapisi

```
models/SGLATrack/data/
├── UAV123/
│   ├── anno/
│   └── data_seq/
├── UAV123_10fps/
│   ├── anno/
│   └── data_seq/
├── uavdt/
│   ├── anno/
│   └── sequences/
├── V4RFlight112/
│   ├── anno/
│   ├── anno_l/
│   ├── data_seq/
│   └── attributes/
├── DTB70/
│   ├── Animal1/
│   └── ...
└── VisDrone2018-SOT-test-dev/
    ├── annotations/
    ├── sequences/
    └── attributes/
```

Test veri setleri: [Baidu Pan](https://pan.baidu.com/s/1MaeGLRcAUbJxksbF_CrOeQ?pwd=5vbv) (kod: 5vbv)

### Analiz

```bash
python tracking/analysis_results.py
# Betik içindeki takipci konfigürasyonlarini ve adlarini düzenlemek gerekir
```

## Model Profilleme

```bash
python tracking/profile_model.py
```

Olcumler:
- **FLOP'lar**: İleri gecis başına kayan nokta islem sayısı
- **Hiz**: Hedef GPU'da saniye başına kare
- **Not**: Makaledeki sonuçlar tek bir RTX 2080Ti üzerinde olculmustur

## Hata Ayiklama Modu

Gorsellestirmeyi etkinlestirmek için `--debug 1` gecin:

- **Visdom olmadan** (`debug=1, use_visdom=False`): Etiketli kareleri `debug/` dizinine kaydeder
- **Visdom ile** (`debug=1, use_visdom=True`): Canli gorsellestirme:
  - Takip sonucu katmani
  - Arama bölgesi kirpmasi
  - Şablon yamasi
  - Ham skor haritası isi haritası
  - Hann pencereli skor haritası isi haritası
  - Aday eleme maskeli arama (CE etkinse)
