# SGLATrack Architecture

[< Back to Index](README.md)

## High-Level Architecture

SGLATrack follows the **one-stream tracking** paradigm. Template and search region images are independently embedded into patch tokens, then concatenated and processed jointly through a transformer backbone. The key innovation is the **Similarity-Guided Layer-Adaptive (SGLA)** mechanism that dynamically selects which transformer layers to execute.

```
Template Image (128x128)  Search Region (256x256)
        |                          |
   Patch Embed (16x16)       Patch Embed (16x16)
        |                          |
   64 tokens + pos_embed     256 tokens + pos_embed
        |                          |
        +--- combine_tokens() ----+
                    |
            320 tokens [B, 320, C]
                    |
        Transformer Layers 0-5 (always execute)
                    |
            Layer 6: MLP predicts active layers
                    |
        Layers 7-11 (selectively execute)
                    |
            recover_tokens()
                    |
        Search tokens [B, 256, C]
                    |
            Prediction Head (CENTER)
                    |
        score_map + size_map + offset_map
                    |
              Bounding Box [cx, cy, w, h]
```

## Backbone Variants

### DeiT-Tiny Distilled (Primary)

Defined in `models/SGLATrack/lib/models/sglatrack/deit.py`.

| Parameter | Value |
|-----------|-------|
| Patch size | 16x16 |
| Embed dimension | 192 |
| Depth | 12 layers |
| Attention heads | 3 |
| MLP ratio | 4 |
| Patch start index | 2 (distillation token) |

Uses Meta's DeiT knowledge distillation approach. The distillation token (index 1) provides additional training signal from a teacher network.

### ViT-Base

Defined in `models/SGLATrack/lib/models/sglatrack/vit.py`.

| Parameter | Value |
|-----------|-------|
| Patch size | 16x16 |
| Embed dimension | 768 |
| Depth | 12 layers |
| Attention heads | 12 |
| MLP ratio | 4 |
| Patch start index | 1 (cls token) |

Standard Vision Transformer with MAE pre-training.

## Similarity-Guided Layer-Adaptive Mechanism

This is the core contribution of SGLATrack, implemented in `models/SGLATrack/lib/models/sglatrack/base_backbone.py`.

### Configuration

```python
enabled_layer_num = 1   # Number of layers to activate beyond start_layer
start_layer = 5         # Layers 0-5 always execute; adaptation begins at layer 6
```

### ThreeLayerMLP — Layer Predictor

```python
class ThreeLayerMLP(nn.Module):
    # input_dim=320 (number of tokens)
    # output_dim=6 (layers 6-11 to choose from)
    fc1: Linear(320, 160)
    relu: ReLU
    fc2: Linear(160, 6)
    sigmoid: Sigmoid  # outputs probabilities per layer
```

The MLP takes the first element of each token's embedding (`x[:,:,0]`) at layer 6 and predicts a probability for each of the remaining 6 layers. `torch.topk()` selects the top-k layers to activate.

### Training Forward Pass

During training (`forward_()` method, lines 129-174):

1. **Layers 0-5**: Execute unconditionally on all batch samples
2. **Layer 6**: Execute, then:
   - Detach intermediate features: `mid = x.detach()`
   - MLP predicts layer probabilities: `pro = self.MLP(x[:,:,0].clone())`
   - Select top-k layers: `topk_indices = torch.topk(pro, enabled_layer_num)`
3. **Layers 7-11**: Execute selectively per batch sample:
   ```python
   idx = torch.where(sorted_topk_indices[:,:] == i)[0]
   if len(idx) > 0:
       x[idx] = blk(x[idx])  # Only process selected samples
   ```
4. **Similarity computation** (with `torch.no_grad()`):
   - Run all remaining layers on detached `mid` features
   - Compute cosine similarity: `cos = F.cosine_similarity(mid, blk(mid))`
   - Build `cos_tensor` — used as the supervision target for the MLP

### Inference Forward Pass

During inference (`forward_test()` method, lines 194-228):

Same layer selection logic, but with **early exit**:

```python
for i, blk in enumerate(self.blocks):
    if i < start_layer:
        x = blk(x)
    elif i == start_layer:
        x = blk(x)
        pro = self.MLP(x[:,:,0].clone())
        sorted_topk_indices = torch.topk(pro, enabled_layer_num) + start_layer + 1
    else:
        idx = torch.where(sorted_topk_indices[:,:] == i)[0]
        if len(idx) > 0:
            x[idx] = blk(x[idx])
            break  # <-- Early exit after first selected layer
```

The `break` statement means the model processes only **7 layers** instead of 12 at inference time (layers 0-6 + 1 selected layer), significantly reducing computation.

## Token Management

Implemented in `models/SGLATrack/lib/models/sglatrack/utils.py`.

### combine_tokens()

Concatenates template and search tokens before transformer processing. Three modes:

| Mode | Description |
|------|-------------|
| `direct` | Simple concatenation: `[template; search]` |
| `template_central` | Search is split in half, template inserted in the middle |
| `partition` | Template is partitioned into windows and prepended |

Default mode is `direct`.

### recover_tokens()

Reverses the concatenation after transformer processing, restoring separate template and search token sequences. Only the **search tokens** are passed to the prediction head.

### Token Dimensions (DeiT-Tiny, default config)

| Region | Image Size | Patch Size | Tokens | Embedding |
|--------|-----------|-----------|--------|-----------|
| Template | 128x128 | 16x16 | 64 | 192-dim |
| Search | 256x256 | 16x16 | 256 | 192-dim |
| **Combined** | — | — | **320** | 192-dim |

## Prediction Heads

Defined in `models/SGLATrack/lib/models/layers/head.py`.

### Center Predictor (Default)

Class `CenterPredictor` — three parallel convolutional branches:

```
Input: [B, C, feat_sz, feat_sz]  (e.g., [B, 192, 16, 16])
        |
   +---------+-----------+
   |         |           |
  Center   Offset      Size
  Branch   Branch      Branch
   |         |           |
score_map  offset_map  size_map
[B,1,16,16] [B,2,16,16] [B,2,16,16]
```

Each branch has 4 conv layers (with BN + ReLU) followed by a final 1x1 conv:
- **Center branch**: Predicts target center location heatmap (sigmoid activated)
- **Offset branch**: Predicts sub-pixel offset from grid cell center (2D)
- **Size branch**: Predicts bounding box width and height (sigmoid activated)

**Bounding box decoding** (`cal_bbox()`):
1. Find peak location in score map: `max_score, idx = torch.max(score_map)`
2. Extract size and offset at peak location
3. Compute box: `cx = (idx_x + offset_x) / feat_sz`, `cy = (idx_y + offset_y) / feat_sz`, `w, h = size`

### Corner Predictor (Alternative)

Class `Corner_Predictor` — two parallel branches predicting top-left and bottom-right corners. Uses **soft-argmax** for differentiable coordinate extraction:

```python
prob_vec = softmax(score_vec)
exp_x = sum(coord_x * prob_vec)
exp_y = sum(coord_y * prob_vec)
```

## Loss Functions

Total loss computed in `models/SGLATrack/lib/train/actors/sglatrack.py`:

```
L_total = 2.0 * L_giou + 5.0 * L_l1 + 1.0 * L_focal + 0.2 * L_cos
```

| Loss | Weight | Purpose | Implementation |
|------|--------|---------|----------------|
| **GIoU** | 2.0 | Bounding box regression (scale-invariant) | `lib/utils/box_ops.py` |
| **L1** | 5.0 | Box coordinate regression (direct supervision) | `nn.L1Loss` |
| **Focal** | 1.0 | Score map localization (handles class imbalance) | `lib/utils/focal_loss.py` |
| **COS (pro_loss)** | 0.2 | Layer selection supervision | L1 between MLP output and similarity target |

### COS Loss Detail

The similarity-guided loss trains the MLP to predict which layers are most useful:

1. **Target computation**: For each layer beyond `start_layer`, compute cosine similarity between input and output features. The layer with the highest similarity (least change) gets a one-hot target.
2. **MLP supervision**: L1 loss between MLP predictions (`pro`) and this one-hot target (`pro_target`).

This teaches the MLP to identify and skip layers that contribute least to the representation — the layer with the highest input-output similarity is the most redundant.

## Candidate Elimination (CE)

Optional mechanism to mask low-confidence regions during training:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CE_START_EPOCH` | 20 | Epoch when CE begins |
| `CE_WARM_EPOCH` | 80 | Warmup duration |
| `CE_LOC` | `[]` (DeiT config) | Layer indices for CE masks |
| `CE_KEEP_RATIO` | `[]` (DeiT config) | Keep ratios per layer |

In the DeiT-Tiny configuration, CE is **disabled** (empty `CE_LOC`).

---

# SGLATrack Mimari

[< İndeks'e Dön](README.md)

## Üst Düzey Mimari

SGLATrack, **tek akışlı takip** paradigmasını takip eder. Şablon ve arama bölgesi görüntüleri bağımsız olarak yama tokenlerine gömülür, ardından birleştirilerek bir transformer omurgasi üzerinden ortaklaşa işlenir. Temel yenilik, hangi transformer katmanlarının çalıştırılacağını dinamik olarak seçen **Benzerlik Rehberli Katman-Uyarlamalı (SGLA)** mekanizmadır.

```
Şablon Görüntü (128x128)    Arama Bölgesi (256x256)
        |                          |
   Yama Gömme (16x16)        Yama Gömme (16x16)
        |                          |
   64 token + konum_gömme    256 token + konum_gömme
        |                          |
        +--- combine_tokens() ----+
                    |
            320 token [B, 320, C]
                    |
        Transformer Katmanlar 0-5 (her zaman çalışır)
                    |
            Katman 6: MLP aktif katmanları tahmin eder
                    |
        Katmanlar 7-11 (seçici çalıştırma)
                    |
            recover_tokens()
                    |
        Arama tokenleri [B, 256, C]
                    |
            Tahmin Basligi (CENTER)
                    |
        skor_haritası + boyut_haritası + ofset_haritası
                    |
              Sinir Kutusu [cx, cy, w, h]
```

## Omurga Varyantları

### DeiT-Tiny Distilled (Birincil)

`models/SGLATrack/lib/models/sglatrack/deit.py` dosyasında tanımlanmıştır.

| Parametre | Değer |
|-----------|-------|
| Yama boyutu | 16x16 |
| Gömme boyutu | 192 |
| Derinlik | 12 katman |
| Dikkat başlıklari | 3 |
| MLP orani | 4 |
| Yama başlangıç indeksi | 2 (distilasyon tokeni) |

Meta'nin DeiT bilgi distilasyonu yaklaşımını kullanır. Distilasyon tokeni (indeks 1), öğretmen ağdan ek eğitim sinyali sağlar.

### ViT-Base

`models/SGLATrack/lib/models/sglatrack/vit.py` dosyasında tanımlanmıştır.

| Parametre | Değer |
|-----------|-------|
| Yama boyutu | 16x16 |
| Gömme boyutu | 768 |
| Derinlik | 12 katman |
| Dikkat başlıklari | 12 |
| MLP orani | 4 |
| Yama başlangıç indeksi | 1 (cls tokeni) |

MAE on-eğitimi ile standart Vision Transformer.

## Benzerlik Rehberli Katman-Uyarlamalı Mekanizma

SGLATrack'in temel katkısı olup `models/SGLATrack/lib/models/sglatrack/base_backbone.py` dosyasında uygulanmıştır.

### Konfigürasyon

```python
enabled_layer_num = 1   # start_layer ötesinde aktive edilecek katman sayısı
start_layer = 5         # Katmanlar 0-5 her zaman çalışır; uyarlama katman 6'da başlar
```

### ThreeLayerMLP — Katman Tahmincisi

```python
class ThreeLayerMLP(nn.Module):
    # input_dim=320 (token sayısı)
    # output_dim=6 (secilecek katmanlar: 6-11)
    fc1: Linear(320, 160)
    relu: ReLU
    fc2: Linear(160, 6)
    sigmoid: Sigmoid  # katman başına olasılık çıkarır
```

MLP, katman 6'da her tokenin gömmesinin ilk elemanını (`x[:,:,0]`) alır ve kalan 6 katmanin her biri için bir olasılık tahmin eder. `torch.topk()` ile aktive edilecek en iyi k katman seçilir.

### Eğitim İleri Geçişi

Eğitim sırasında (`forward_()` metodu, satırlar 129-174):

1. **Katmanlar 0-5**: Tüm yığıt örnekleri üzerinde koşulsuz çalıştırılır
2. **Katman 6**: Çalıştırılır, ardından:
   - Ara özellikler ayrılır: `mid = x.detach()`
   - MLP katman olasılıklerini tahmin eder: `pro = self.MLP(x[:,:,0].clone())`
   - En iyi k katman seçilir: `topk_indices = torch.topk(pro, enabled_layer_num)`
3. **Katmanlar 7-11**: Yigit örneği bazinda seçici olarak çalıştırılır:
   ```python
   idx = torch.where(sorted_topk_indices[:,:] == i)[0]
   if len(idx) > 0:
       x[idx] = blk(x[idx])  # Yalnızca seçilen örnekler işlenir
   ```
4. **Benzerlik hesaplama** (`torch.no_grad()` ile):
   - Ayrılmış `mid` özellikleri üzerinde tüm kalan katmanlar çalıştırılır
   - Kosinüs benzerliği hesaplanır: `cos = F.cosine_similarity(mid, blk(mid))`
   - `cos_tensor` oluşturulur — MLP için denetim hedefi olarak kullanılır

### Çıkarım İleri Geçişi

Çıkarım sırasında (`forward_test()` metodu, satırlar 194-228):

Aynı katman seçim mantığı, ancak **erken çıkış** ile:

```python
for i, blk in enumerate(self.blocks):
    if i < start_layer:
        x = blk(x)
    elif i == start_layer:
        x = blk(x)
        pro = self.MLP(x[:,:,0].clone())
        sorted_topk_indices = torch.topk(pro, enabled_layer_num) + start_layer + 1
    else:
        idx = torch.where(sorted_topk_indices[:,:] == i)[0]
        if len(idx) > 0:
            x[idx] = blk(x[idx])
            break  # <-- Ilk seçilen katmandan sonra erken çıkış
```

`break` ifadesi, modelin çıkarım zamanında 12 yerine yalnızca **7 katman** işlemesi anlamına gelir (katmanlar 0-6 + 1 seçilen katman), hesaplama maliyetini önemli ölçüde azaltır.

## Token Yönetimi

`models/SGLATrack/lib/models/sglatrack/utils.py` dosyasında uygulanmıştır.

### combine_tokens()

Transformer işlemesinden önce şablon ve arama tokenlerini birleştirir. Üç mod:

| Mod | Açıklama |
|-----|----------|
| `direct` | Basit birleştirme: `[şablon; arama]` |
| `template_central` | Arama ikiye bölünür, şablon ortaya yerleştirilir |
| `partition` | Şablon pencerelere bölünüp başına eklenir |

Varsayılan mod `direct`'tir.

### recover_tokens()

Transformer işlemesinden sonra birleştirmeyi tersine çevirir, ayri şablon ve arama token dizilerini geri yükler. Yalnızca **arama tokenleri** tahmin basligina iletilir.

### Token Boyutları (DeiT-Tiny, varsayılan konfigürasyon)

| Bölge | Görüntü Boyutu | Yama Boyutu | Token | Gömme |
|-------|---------------|-------------|-------|-------|
| Şablon | 128x128 | 16x16 | 64 | 192-boyut |
| Arama | 256x256 | 16x16 | 256 | 192-boyut |
| **Birleşik** | — | — | **320** | 192-boyut |

## Tahmin Başlıklari

`models/SGLATrack/lib/models/layers/head.py` dosyasında tanımlanmıştır.

### Merkez Tahmincisi (Varsayılan)

`CenterPredictor` sınıfı — üç paralel evrişimsel dal:

```
Giriş: [B, C, feat_sz, feat_sz]  (orn., [B, 192, 16, 16])
        |
   +---------+-----------+
   |         |           |
  Merkez   Ofset       Boyut
  Dali     Dali        Dali
   |         |           |
skor_haritası ofset_haritası boyut_haritası
[B,1,16,16]  [B,2,16,16]   [B,2,16,16]
```

Her dalda 4 evrişim katmani (BN + ReLU ile) ve son 1x1 evrişim bulunur:
- **Merkez dali**: Hedef merkez konumu isi haritasını tahmin eder (sigmoid aktive)
- **Ofset dali**: Izgara hücre merkezinden alt-piksel ofseti tahmin eder (2B)
- **Boyut dali**: Sinir kutusu genişlik ve yüksekliğini tahmin eder (sigmoid aktive)

**Sinir kutusu çözümleme** (`cal_bbox()`):
1. Skor haritasında tepe noktasını bul: `max_score, idx = torch.max(score_map)`
2. Tepe noktasında boyut ve ofseti çıkar
3. Kutuyu hesapla: `cx = (idx_x + offset_x) / feat_sz`, `cy = (idx_y + offset_y) / feat_sz`, `w, h = size`

### Köşe Tahmincisi (Alternatif)

`Corner_Predictor` sınıfı — sol-üst ve sağ-alt köşeleri tahmin eden iki paralel dal. Türevlenebilir koordinat çıkarımi için **yumuşak-argmax** kullanır:

```python
prob_vec = softmax(score_vec)
exp_x = sum(coord_x * prob_vec)
exp_y = sum(coord_y * prob_vec)
```

## Kayıp Fonksiyonları

Toplam kayıp `models/SGLATrack/lib/train/actors/sglatrack.py` dosyasında hesaplanır:

```
L_toplam = 2.0 * L_giou + 5.0 * L_l1 + 1.0 * L_focal + 0.2 * L_cos
```

| Kayıp | Ağırlık | Amaç | Uygulama |
|-------|---------|------|----------|
| **GIoU** | 2.0 | Sinir kutusu regresyonu (ölçek bağımsız) | `lib/utils/box_ops.py` |
| **L1** | 5.0 | Kutu koordinat regresyonu (doğrudan denetim) | `nn.L1Loss` |
| **Focal** | 1.0 | Skor haritası lokalizasyonu (sınıf dengesizliğini yönetir) | `lib/utils/focal_loss.py` |
| **COS (pro_loss)** | 0.2 | Katman seçimi denetimi | MLP çıkışi ile benzerlik hedefi arasi L1 |

### COS Kaybi Detayı

Benzerlik rehberli kayıp, MLP'yi hangi katmanların en faydalı olduğunu tahmin etmesi için egitir:

1. **Hedef hesaplama**: `start_layer` ötesindeki her katman icin, giriş ve çıkış özellikleri arasındaki kosinüs benzerliği hesaplanır. En yüksek benzerliğe (en az değişikliğe) sahip katman tek-sıcak hedef alır.
2. **MLP denetimi**: MLP tahminleri (`pro`) ile bu tek-sıcak hedef (`pro_target`) arasında L1 kaybi.

Bu, MLP'ye temsile en az katkıda bulunan katmanları tanımayı ve atlamayı öğretir — en yüksek giriş-çıkış benzerliğine sahip katman en gereksiz olandır.

## Aday Eleme (CE)

Eğitim sırasında düşük güvenilirlikli bölgeleri maskelemek için isteğe bağlı mekanizma:

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `CE_START_EPOCH` | 20 | CE'nin başladığı epoch |
| `CE_WARM_EPOCH` | 80 | Isınma süresi |
| `CE_LOC` | `[]` (DeiT konfig) | CE maskeleri için katman indeksleri |
| `CE_KEEP_RATIO` | `[]` (DeiT konfig) | Katman başına tutma oranları |

DeiT-Tiny konfigürasyonunda CE **devre dışıdır** (boş `CE_LOC`).
