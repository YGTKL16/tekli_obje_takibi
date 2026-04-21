# SGLATrack Integration with Tracker Project

[< Back to Index](README.md)

## Overview

The Tracker project integrates SGLATrack as its AI detection backbone within a hybrid tracking pipeline:

```
SGLATrackWrapper  →  DecisionMaker  →  Kalman Filter (C++)  →  Visualizer
     (AI)           (gate logic)        (state estimation)     (display)
```

The AI model provides per-frame bounding box predictions and confidence scores. The DecisionMaker decides whether to trust the AI output or let the Kalman Filter coast. This hybrid approach enables robust tracking through occlusions and low-confidence scenarios.

## SGLATrackWrapper API

Defined in `python/tracker/sglatrack_wrapper.py`.

### Constructor

```python
wrapper = SGLATrackWrapper(checkpoint_path=None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_path` | `models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar` | Path to model weights |

Model loading is **deferred** (lazy) — the network is only loaded on the first call to `init()`. This avoids GPU memory allocation until the tracker is actually used.

### init(frame, bbox)

```python
wrapper.init(frame: np.ndarray, bbox: np.ndarray)
```

Initializes the tracker with the first frame and target bounding box.

| Parameter | Type | Format |
|-----------|------|--------|
| `frame` | `np.ndarray` | RGB image `(H, W, 3)` uint8 |
| `bbox` | `np.ndarray` | `[x, y, w, h]` **top-left** format |

Internally:
1. Loads model if not already loaded (`_load_model()`)
2. Extracts template patch with `sample_target(frame, bbox, factor=2.0, output_sz=128)`
3. Preprocesses template (ImageNet normalization, CUDA transfer)
4. Generates CE mask if applicable (disabled in DeiT config)

### track(frame)

```python
bbox, confidence = wrapper.track(frame: np.ndarray)
```

Tracks the target in the given frame.

| Parameter | Type | Format |
|-----------|------|--------|
| `frame` | `np.ndarray` | RGB image `(H, W, 3)` uint8 |

**Returns**:

| Value | Type | Format |
|-------|------|--------|
| `bbox` | `np.ndarray` | `[x, y, w, h]` **top-left** format, float32 |
| `confidence` | `float` | Peak response value in `[0, 1]` |

Processing steps:
1. Extract search region: `sample_target(frame, prev_state, factor=4.0, output_sz=256)`
2. Preprocess and run network forward pass
3. Apply Hann window to score map
4. Decode bounding box from response
5. Map coordinates back to original image space
6. Clip to image boundaries (margin=10)
7. Update internal state

## Pipeline Integration

The full pipeline is orchestrated in `python/tracker/pipeline.py` (`Pipeline` class).

### Per-Frame Flow

```python
# 1. AI inference (~10-15 ms)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
ai_bbox, confidence = self.ai.track(frame_rgb)

# 2. Decision gate (~0.01 ms)
kf_state = np.array(self.kf.get_state()).flatten()
do_update = self.decision.should_update(confidence, ai_bbox, kf_state)
track_state = self.state_machine.step(confidence)

# 3. Kalman filter (~0.05 ms)
if do_update and track_state == tracker_cpp.TrackState.TRACKING:
    state = np.array(self.kf.update(ai_bbox)).flatten()
else:
    state = np.array(self.kf.predict()).flatten()

# 4. Visualize (~1-2 ms)
vis = visualize(frame, ai_bbox, state, confidence, state_name, fps, coast_count)
```

### Performance Budget

| Stage | Target | Description |
|-------|--------|-------------|
| Frame read | 1-3 ms | OpenCV VideoCapture |
| AI inference | 10-15 ms | SGLATrack forward pass |
| Decision | 0.01 ms | Confidence + IoU check |
| Kalman filter | 0.05 ms | C++ predict/update |
| Visualization | 1-2 ms | OpenCV drawing |
| **Total** | **<= 30 ms** | **Real-time target** |

Budget check runs every 100 frames and logs timing statistics.

## DecisionMaker

Defined in `python/tracker/decision.py`.

### Logic

```python
def should_update(confidence, ai_bbox, kf_predicted) -> bool:
    # Gate 1: Confidence threshold
    if confidence < 0.3:
        return False  # → coast
    
    # Gate 2: IoU cross-check (outlier rejection)
    kf_bbox = kf_predicted[:4]
    iou = compute_iou(ai_bbox, kf_bbox)
    if kf_is_initialized and iou < 0.2:
        return False  # → coast (AI prediction too far from KF)
    
    return True  # → update KF with AI measurement
```

### Thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| `confidence_threshold` | 0.3 | AI predictions below this are ignored |
| `iou_threshold` | 0.2 | AI bbox must overlap KF prediction by at least 20% |

## State Machine

The C++ `TrackerState` class manages three tracking states:

```
         confidence >= threshold
LOST ←——————————————————————————→ TRACKING
  ↑                                    |
  |  coast_count > max_coast_frames    | confidence < threshold
  |                                    ↓
  +←—————————————————————————————— COASTING
```

| State | Behavior | KF Action |
|-------|----------|-----------|
| **TRACKING** | AI is reliable | `predict()` + `update(measurement)` |
| **COASTING** | AI lost, up to 60 frames | `predict()` only (motion model) |
| **LOST** | Coasting exceeded limit | Tracking terminated |

## Configuration

All integration parameters are in `configs/tracker_config.yaml`:

```yaml
kalman:
  process_noise:
    position: 1.0          # x, y noise
    size: 1.0              # w, h noise
    velocity: 0.01         # vx, vy noise
    size_velocity: 0.0001  # vw, vh noise
  measurement_noise:
    position: 1.0          # x, y measurement noise
    size: 10.0             # w, h measurement noise

decision:
  confidence_threshold: 0.3
  iou_threshold: 0.2

coasting:
  max_frames: 60

sglatrack:
  checkpoint: "models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar"
  config_yaml: "models/SGLATrack/experiments/sglatrack/deit_distilled.yaml"
```

## Setup Steps

### 1. Install Dependencies

```bash
source scripts/setup_env.sh
```

### 2. Build C++ Components

```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

### 2b. Open-Source High-Assurance Toolchains

For an open-source hardened build path based on GCC/Clang and Rust, see
[Open-Source High-Assurance Toolchains](safety_toolchains.md).

### 3. Download Model Checkpoint

Download from [Google Drive](https://drive.google.com/drive/folders/1sHL7aFVZFwkPy6js48x-EKfoZC7oJc9X?usp=sharing) and place at:
```
models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar
```

### 4. Run Pipeline

```python
from python.tracker.pipeline import Pipeline
import numpy as np

pipeline = Pipeline(
    video_path="data/contest_release/dataset1/Car_video/Car_video.mp4",
    initial_bbox=np.array([x, y, w, h], dtype=np.float32),
    model_path="models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar",
)
pipeline.run()
```

## BBox Format Convention

Throughout the Tracker project, `SGLATrackWrapper` uses **top-left `[x, y, w, h]`** format:
- `x`: left edge of bounding box
- `y`: top edge of bounding box
- `w`: width
- `h`: height

The Kalman Filter state uses the same **top-left `[x, y, w, h, vx, vy, vw, vh]`** convention at the integration boundary.

## TensorRT Acceleration (Optional)

For competition deployment, TensorRT optimization is configured in `tracker_config.yaml`:

```yaml
tensorrt:
  onnx_path: "models/sglatrack.onnx"
  engine_path: "models/sglatrack_fp16.engine"
  fp16: true
```

---

# Tracker Projesiyle SGLATrack Entegrasyonu

[< İndeks'e Dön](README.md)

## Genel Bakış

Tracker projesi, SGLATrack'i hibrit takip pipeline'inda AI algilama omurgasi olarak entegre eder:

```
SGLATrackWrapper  →  DecisionMaker  →  Kalman Filtresi (C++)  →  Gorsellestiirci
     (AI)            (gecit mantığı)     (durum tahmini)          (gosterim)
```

AI modeli, kare başına sinir kutusu tahminleri ve güven skorlari sağlar. DecisionMaker, AI ciktisina güvenilip güvenilmeyecegine veya Kalman Filtresinin devam edip etmeyecegine karar verir. Bu hibrit yaklasim, kapanmalar ve düşük güvenilirlik senaryolarinda saglikli takip sağlar.

## SGLATrackWrapper API

`python/tracker/sglatrack_wrapper.py` dosyasında tanımlanmıştır.

### Yapici

```python
wrapper = SGLATrackWrapper(checkpoint_path=None)
```

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `checkpoint_path` | `models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar` | Model ağırlıkları yolu |

Model yukleme **ertelenmistir** (tembel) — ag yalnızca `init()` için ilk cagri sırasında yuklenir. Bu, takipci gercekten kullanılana kadar GPU bellek tahsisini onler.

### init(kare, kutu)

```python
wrapper.init(kare: np.ndarray, kutu: np.ndarray)
```

Takipciyi ilk kare ve hedef sinir kutusuyla baslatir.

| Parametre | Tür | Format |
|-----------|-----|--------|
| `kare` | `np.ndarray` | RGB görüntü `(H, W, 3)` uint8 |
| `kutu` | `np.ndarray` | `[x, y, w, h]` **sol-üst** format |

Dahili olarak:
1. Henüz yuklenmemisse modeli yükler (`_load_model()`)
2. `sample_target(kare, kutu, faktor=2.0, çıkış_boyutu=128)` ile şablon yamasi çıkarır
3. Şablonu on isler (ImageNet normalizasyonu, CUDA aktarimi)
4. Uygulanabilirse CE maskesi uretir (DeiT konfigürasyonunda devre disi)

### track(kare)

```python
kutu, güven = wrapper.track(kare: np.ndarray)
```

Verilen karede hedefi takip eder.

| Parametre | Tür | Format |
|-----------|-----|--------|
| `kare` | `np.ndarray` | RGB görüntü `(H, W, 3)` uint8 |

**Donüs değerleri**:

| Değer | Tür | Format |
|-------|-----|--------|
| `kutu` | `np.ndarray` | `[x, y, w, h]` **sol-üst** format, float32 |
| `güven` | `float` | Tepe yanit değeri `[0, 1]` araliginda |

Isleme adimlari:
1. Arama bölgesi çıkar: `sample_target(kare, onceki_durum, faktor=4.0, çıkış_boyutu=256)`
2. On isle ve ag ileri gecisini calistir
3. Skor haritasına Hann penceresi uygula
4. Yanittan sinir kutusu coz
5. Koordinatlari orijinal görüntü uzayina geri esle
6. Görüntü sinrlarina kirp (marj=10)
7. Dahili durumu güncelle

## Pipeline Entegrasyonu

Tam pipeline `python/tracker/pipeline.py` dosyasında (`Pipeline` sınıfı) yönetilir.

### Kare Basina Akis

```python
# 1. AI çıkarımi (~10-15 ms)
kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
ai_kutu, güven = self.ai.track(kare_rgb)

# 2. Karar gecidi (~0.01 ms)
kf_durum = np.array(self.kf.get_state()).flatten()
güncelle = self.decision.should_update(güven, ai_kutu, kf_durum)
takip_durumu = self.state_machine.step(güven)

# 3. Kalman filtresi (~0.05 ms)
if güncelle and takip_durumu == tracker_cpp.TrackState.TRACKING:
    durum = np.array(self.kf.update(ai_kutu)).flatten()
else:
    durum = np.array(self.kf.predict()).flatten()

# 4. Gorsellestirme (~1-2 ms)
vis = visualize(kare, ai_kutu, durum, güven, durum_adi, fps, sahil_sayaci)
```

### Performans Bütcesi

| Asama | Hedef | Açıklama |
|-------|-------|----------|
| Kare okuma | 1-3 ms | OpenCV VideoCapture |
| AI çıkarımi | 10-15 ms | SGLATrack ileri gecisi |
| Karar | 0.01 ms | Güven + IoU kontrolü |
| Kalman filtresi | 0.05 ms | C++ tahmin/güncelleme |
| Gorsellestirme | 1-2 ms | OpenCV cizim |
| **Toplam** | **<= 30 ms** | **Gercek zamanli hedef** |

Bütce kontrolü her 100 karede çalışır ve zamanlama istatistiklerini günlükler.

## Karar Verici (DecisionMaker)

`python/tracker/decision.py` dosyasında tanımlanmıştır.

### Mantik

```python
def should_update(güven, ai_kutu, kf_tahmini) -> bool:
    # Gecit 1: Güven esigi
    if güven < 0.3:
        return False  # → sahil modu
    
    # Gecit 2: IoU capraz kontrol (aykiri değer reddi)
    kf_kutu = kf_tahmini[:4]
    iou = compute_iou(ai_kutu, kf_kutu)
    if kf_baslatilmis and iou < 0.2:
        return False  # → sahil modu (AI tahmini KF'den cok uzak)
    
    return True  # → KF'yi AI olcumuyle güncelle
```

### Esik Değerleri

| Parametre | Değer | Etki |
|-----------|-------|------|
| `confidence_threshold` | 0.3 | Bu değerin altindaki AI tahminleri yok sayilir |
| `iou_threshold` | 0.2 | AI kutusu KF tahminiyle en az %20 örtüsmeli |

## Durum Makinesi

C++ `TrackerState` sınıfı üç takip durumunu yönetir:

```
         güven >= esik
KAYIP ←——————————————————————————→ TAKIP
  ↑                                    |
  | sahil_sayaci > maks_sahil_kareler  | güven < esik
  |                                    ↓
  +←—————————————————————————————— SAHIL
```

| Durum | Davranis | KF Eylemi |
|-------|----------|-----------|
| **TAKIP** | AI güvenilir | `predict()` + `update(olcum)` |
| **SAHIL** | AI kayıp, 60 kareye kadar | Yalnızca `predict()` (hareket modeli) |
| **KAYIP** | Sahil limiti asildi | Takip sonlandirildi |

## Konfigürasyon

Tüm entegrasyon parametreleri `configs/tracker_config.yaml` dosyasındadir:

```yaml
kalman:
  process_noise:
    position: 1.0          # x, y gürültüsü
    size: 1.0              # w, h gürültüsü
    velocity: 0.01         # vx, vy gürültüsü
    size_velocity: 0.0001  # vw, vh gürültüsü
  measurement_noise:
    position: 1.0          # x, y olcum gürültüsü
    size: 10.0             # w, h olcum gürültüsü

decision:
  confidence_threshold: 0.3
  iou_threshold: 0.2

coasting:
  max_frames: 60

sglatrack:
  checkpoint: "models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar"
  config_yaml: "models/SGLATrack/experiments/sglatrack/deit_distilled.yaml"
```

## Kurulum Adimlari

### 1. Bagimliliklari Kur

```bash
source scripts/setup_env.sh
```

### 2. C++ Bilesenlerini Derle

```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

### 3. Model Kontrol Noktasini Indir

[Google Drive](https://drive.google.com/drive/folders/1sHL7aFVZFwkPy6js48x-EKfoZC7oJc9X?usp=sharing) adresinden indirip su konuma yerlestirin:
```
models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar
```

### 4. Pipeline'i Calistir

```python
from python.tracker.pipeline import Pipeline
import numpy as np

pipeline = Pipeline(
    video_path="data/contest_release/dataset1/Car_video/Car_video.mp4",
    initial_bbox=np.array([x, y, w, h], dtype=np.float32),
    model_path="models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar",
)
pipeline.run()
```

## Sinir Kutusu Format Kurali

Tracker projesi genelinde `SGLATrackWrapper` **sol-üst `[x, y, w, h]`** formati kullanır:
- `x`: sinir kutusunun sol kenari
- `y`: sinir kutusunun üst kenari
- `w`: genişlik
- `h`: yükseklik

Kalman Filtresi durumu da entegrasyon sinirinda ayni **sol-üst `[x, y, w, h, vx, vy, vw, vh]`** düzenini kullanir.

## TensorRT Hizlandirma (Istege Bagli)

Yarisma dagitimi için TensorRT optimizasyonu `tracker_config.yaml` dosyasında konfigüre edilmiştir:

```yaml
tensorrt:
  onnx_path: "models/sglatrack.onnx"
  engine_path: "models/sglatrack_fp16.engine"
  fp16: true
```
