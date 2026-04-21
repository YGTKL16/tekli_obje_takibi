# SGLATrack Performance Benchmarks

[< Back to Index](README.md)

## Aerial Dataset Results

Primary benchmarks for UAV tracking scenarios:

| Tracker | UAV123 (AUC) | UAV123_10FPS (AUC) | UAVDT (AUC) | DTB70 (AUC) | UAVTrack112 (AUC) | UAVTrack_L (AUC) |
|:-------:|:------------:|:-------------------:|:-----------:|:-----------:|:-----------------:|:----------------:|
| **SGLATrack-DeiT*** | **66.9** | **65.5** | 59.9 | 65.1 | **67.5** | 64.0 |
| SGLATrack-ViT | 66.1 | 64.5 | **60.0** | **65.8** | 67.3 | **64.3** |
| SGLATrack-EVA | 65.1 | 64.3 | 57.9 | 63.8 | 66.9 | 64.7 |

**Key observations**:
- DeiT variant leads on most aerial benchmarks despite being the smallest model
- ViT variant is competitive and slightly better on UAVDT and DTB70
- EVA variant underperforms relative to its model size

## Generic Dataset Results

Cross-domain tracking benchmarks:

| Tracker | TrackingNet (AUC) | LaSOT (AUC) | GOT-10k (AO) |
|:-------:|:-----------------:|:-----------:|:------------:|
| **SGLATrack-DeiT*** | **79.5** | 63.0 | **66.3** |
| SGLATrack-ViT | 79.4 | **64.1** | 66.0 |
| SGLATrack-EVA | 77.7 | 60.9 | 64.2 |

**Key observations**:
- DeiT achieves best performance on TrackingNet and GOT-10k
- ViT leads on LaSOT (long-term sequences benefit from larger model capacity)
- All variants demonstrate strong generalization across dataset types

## Metrics Explained

### AUC (Area Under the Curve)

Used for UAV123, UAVDT, DTB70, UAVTrack, VisDrone, TrackingNet, and LaSOT.

The success plot measures the percentage of frames where the IoU (Intersection over Union) between the predicted and ground truth bounding box exceeds a threshold. AUC is the area under this curve as the threshold varies from 0 to 1.

- **Higher is better**
- Range: 0-100 (reported as percentage)
- An AUC of 66.9 means the tracker maintains good overlap across most threshold levels

### AO (Average Overlap)

Used for GOT-10k.

The average of IoU values across all frames and sequences in the test set.

- **Higher is better**
- Range: 0-100 (reported as percentage)
- Provides a single-number summary of tracking accuracy

## Variant Comparison

### DeiT-Tiny Distilled (Recommended)

| Aspect | Detail |
|--------|--------|
| **Strengths** | Best speed/accuracy trade-off, lightweight, distillation helps generalization |
| **Weaknesses** | Slightly lower on long-term benchmarks (LaSOT) |
| **Best for** | Real-time UAV tracking, resource-constrained deployment |
| **Embed dim** | 192 |
| **Depth** | 12 layers (7 effective at inference with SGLA) |

### ViT-Base

| Aspect | Detail |
|--------|--------|
| **Strengths** | Best LaSOT performance, higher model capacity |
| **Weaknesses** | Slower, larger memory footprint |
| **Best for** | Offline analysis, high-accuracy requirements |
| **Embed dim** | 768 |
| **Depth** | 12 layers (7 effective at inference with SGLA) |

### EVA

| Aspect | Detail |
|--------|--------|
| **Strengths** | Advanced pre-training approach |
| **Weaknesses** | Lowest benchmark scores among the three variants |
| **Best for** | Research exploration |

## Layer-Adaptive Speed Gain

The SGLA mechanism reduces inference cost by executing only 7 of 12 transformer layers:

```
Standard ViT:     Layers 0-11 → 12 layers computed
SGLATrack:        Layers 0-6 + 1 selected → 7 layers computed
                  Reduction: ~42% fewer layer computations
```

The MLP layer predictor and similarity computation add negligible overhead compared to the saved transformer layer computations.

**Effective computation at inference**:
- Layers 0-5: Always executed (6 layers)
- Layer 6: Always executed + MLP prediction (1 layer + MLP)
- Layers 7-11: Only 1 of 5 executed (1 layer, then early exit)
- **Total: 7 transformer layers + 1 lightweight MLP**

## Profiling

To measure FLOPs and speed on your hardware:

```bash
cd models/SGLATrack
python tracking/profile_model.py
```

**Reference speed** (from the paper): Measured on a single NVIDIA RTX 2080Ti GPU.

## Evaluation Commands

Run evaluation on specific datasets:

```bash
# Aerial datasets
python tracking/test.py sglatrack deit_distilled --dataset_name uav123 --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uav123_10fps --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uavdt --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name dtb70 --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uavtrack112 --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uavtrack --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name visdrone --threads 8 --num_gpus 4

# Analyze results
python tracking/analysis_results.py
```

---

# SGLATrack Performans Kıyaslamalari

[< İndeks'e Dön](README.md)

## Hava Veri Seti Sonuçlari

IHA takip senaryolari için birincil kıyaslamalar:

| Takipci | UAV123 (AUC) | UAV123_10FPS (AUC) | UAVDT (AUC) | DTB70 (AUC) | UAVTrack112 (AUC) | UAVTrack_L (AUC) |
|:-------:|:------------:|:-------------------:|:-----------:|:-----------:|:-----------------:|:----------------:|
| **SGLATrack-DeiT*** | **66.9** | **65.5** | 59.9 | 65.1 | **67.5** | 64.0 |
| SGLATrack-ViT | 66.1 | 64.5 | **60.0** | **65.8** | 67.3 | **64.3** |
| SGLATrack-EVA | 65.1 | 64.3 | 57.9 | 63.8 | 66.9 | 64.7 |

**Temel gozlemler**:
- DeiT varyanti en kücük model olmasina ragmen cogu hava kiyaslamasinda lider
- ViT varyanti rekabetci ve UAVDT ile DTB70'te hafifce daha iyi
- EVA varyanti model boyutuna gore düşük performans gosteriyor

## Genel Veri Seti Sonuçlari

Alan ötesi takip kıyaslamalari:

| Takipci | TrackingNet (AUC) | LaSOT (AUC) | GOT-10k (AO) |
|:-------:|:-----------------:|:-----------:|:------------:|
| **SGLATrack-DeiT*** | **79.5** | 63.0 | **66.3** |
| SGLATrack-ViT | 79.4 | **64.1** | 66.0 |
| SGLATrack-EVA | 77.7 | 60.9 | 64.2 |

**Temel gozlemler**:
- DeiT, TrackingNet ve GOT-10k'da en iyi performansi elde ediyor
- ViT, LaSOT'ta lider (uzun vadeli sekanslar büyük model kapasitesinden faydalaniyor)
- Tüm varyantlar veri seti türleri arasında güclü genelleme gosteriyor

## Metrik Açıklamalari

### AUC (Egri Altinda Kalan Alan)

UAV123, UAVDT, DTB70, UAVTrack, VisDrone, TrackingNet ve LaSOT için kullanılır.

Basari grafigi, tahmin edilen ve gercek sinir kutusu arasındaki IoU'nun (Kesisim/Birlesim Orani) bir esigi astigi karelerin yüzdesini olcer. AUC, esik 0'dan 1'e degisirken bu egrinin altinda kalan alandir.

- **Yüksek olan daha iyi**
- Aralik: 0-100 (yüzde olarak raporlanir)
- 66.9'luk bir AUC, takipcinin cogu esik seviyesinde iyi örtüsmeyi korudugu anlamına gelir

### AO (Ortalama Örtüsme)

GOT-10k için kullanılır.

Test setindeki tüm kareler ve sekanslar boyunca IoU değerlerinin ortalamasi.

- **Yüksek olan daha iyi**
- Aralik: 0-100 (yüzde olarak raporlanir)
- Takip dogrulugu için tek sayili bir ozet sağlar

## Varyant Karsilastirmasi

### DeiT-Tiny Distilled (Önerilen)

| Ozellik | Detay |
|---------|-------|
| **Güclü yanlari** | En iyi hiz/dogruluk dengesi, hafif, distilasyon genellemeye yardimci |
| **Zayif yanlari** | Uzun vadeli kıyaslamalarda (LaSOT) biraz düşük |
| **En iyi kullanim** | Gercek zamanli IHA takibi, kaynak kısıtlı dagitim |
| **Gömme boyutu** | 192 |
| **Derinlik** | 12 katman (SGLA ile çıkarımda 7 etkin) |

### ViT-Base

| Ozellik | Detay |
|---------|-------|
| **Güclü yanlari** | En iyi LaSOT performansi, yüksek model kapasitesi |
| **Zayif yanlari** | Daha yavas, büyük bellek ayak izi |
| **En iyi kullanim** | Cevrimdisi analiz, yüksek dogruluk gereksinimleri |
| **Gömme boyutu** | 768 |
| **Derinlik** | 12 katman (SGLA ile çıkarımda 7 etkin) |

### EVA

| Ozellik | Detay |
|---------|-------|
| **Güclü yanlari** | Gelişmiş on-eğitim yaklaşımı |
| **Zayif yanlari** | Üç varyant arasında en düşük kiyaslama skorlari |
| **En iyi kullanim** | Arastirma kesfi |

## Katman-Uyarlamalı Hiz Kazanimi

SGLA mekanizmasi, 12 transformer katmanindan yalnızca 7'sini calistirarak çıkarım maliyetini azaltır:

```
Standart ViT:     Katmanlar 0-11 → 12 katman hesaplanır
SGLATrack:        Katmanlar 0-6 + 1 seçilen → 7 katman hesaplanır
                  Azaltma: ~%42 daha az katman hesaplamasi
```

MLP katman tahmincisi ve benzerlik hesaplamasi, kaydedilen transformer katman hesaplamalarina kiyasla ihmal edilebilir yük ekler.

**Çıkarımda etkin hesaplama**:
- Katmanlar 0-5: Her zaman çalıştırılır (6 katman)
- Katman 6: Her zaman çalıştırılır + MLP tahmini (1 katman + MLP)
- Katmanlar 7-11: 5'ten yalnızca 1'i çalıştırılır (1 katman, sonra erken çıkış)
- **Toplam: 7 transformer katmani + 1 hafif MLP**

## Profilleme

Donanim üzerinde FLOP ve hiz olcmek icin:

```bash
cd models/SGLATrack
python tracking/profile_model.py
```

**Referans hiz** (makaleden): Tek bir NVIDIA RTX 2080Ti GPU üzerinde olculmustur.

## Değerlendirme Komutlari

Belirli veri setlerinde değerlendirme calistirin:

```bash
# Hava veri setleri
python tracking/test.py sglatrack deit_distilled --dataset_name uav123 --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uav123_10fps --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uavdt --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name dtb70 --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uavtrack112 --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name uavtrack --threads 8 --num_gpus 4
python tracking/test.py sglatrack deit_distilled --dataset_name visdrone --threads 8 --num_gpus 4

# Sonuçlari analiz et
python tracking/analysis_results.py
```
