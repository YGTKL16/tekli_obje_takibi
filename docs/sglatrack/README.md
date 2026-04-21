# SGLATrack Technical Documentation

> **Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking**
>
> Chaocan Xue, Bineng Zhong, Qihua Liang, Yaozong Zheng, Ning Li, Yuanliang Xue, Shuxiang Song
>
> **CVPR 2025** | [Paper (arXiv)](https://arxiv.org/abs/2503.06625) | [GitHub](https://github.com/GXNU-ZhongLab/SGLATrack) | [Models](https://drive.google.com/drive/folders/1sHL7aFVZFwkPy6js48x-EKfoZC7oJc9X?usp=sharing) | [Raw Results](https://drive.google.com/drive/folders/1ss-KQqPsfIXeOcl_h3w6Q09dEk07DjUy?usp=sharing)

## Overview

SGLATrack (**S**imilarity-**G**uided **L**ayer-**A**daptive Tracker) is a vision transformer-based single-object tracker designed for UAV tracking scenarios. Its core innovation is a **layer-adaptive mechanism** that dynamically selects which transformer layers to execute during inference, guided by cosine similarity between intermediate features. This reduces redundant computation while maintaining tracking accuracy — a critical requirement for resource-constrained UAV platforms.

The model builds on the one-stream tracking paradigm (OSTrack / AVTrack), where template and search region tokens are concatenated and processed jointly through a vision transformer backbone. A lightweight MLP predicts which of the deeper transformer layers should be activated, enabling early exit at inference time.

### Model Variants

| Variant | Backbone | Embed Dim | Depth | Heads | Params |
|---------|----------|-----------|-------|-------|--------|
| **SGLATrack-DeiT*** | DeiT-Tiny Distilled | 192 | 12 | 3 | Lightweight |
| SGLATrack-ViT | ViT-Base | 768 | 12 | 12 | Standard |
| SGLATrack-EVA | EVA | — | — | — | Advanced |

\* Primary variant used in this project.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Model architecture, layer-adaptive mechanism, prediction heads, loss functions |
| [Training](training.md) | Training pipeline, datasets, configuration, distributed training |
| [Inference](inference.md) | Tracking pipeline, frame-by-frame processing, evaluation |
| [Integration](integration.md) | Integration with the Tracker project, API reference, pipeline flow |
| [Open-Source High-Assurance Toolchains](safety_toolchains.md) | GCC/Clang and Rust toolchain integration for hardened builds |
| [Benchmarks](benchmarks.md) | Performance results on aerial and generic datasets |

## Quick Start

### 1. Set Up Paths

```bash
cd models/SGLATrack
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

### 2. Download Pre-trained Weights

```bash
# DeiT-Tiny Distilled backbone
wget https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth \
  -P models/SGLATrack/pretrained_models/

# SGLATrack checkpoint (from Google Drive)
# Place at: models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar
```

### 3. Run Tracking (within Tracker project)

```python
from python.tracker.sglatrack_wrapper import SGLATrackWrapper

tracker = SGLATrackWrapper()
tracker.init(first_frame, initial_bbox)  # [x, y, w, h] top-left
bbox, confidence = tracker.track(next_frame)
```

## Citation

```bibtex
@inproceedings{sglatrack,
  title={Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking},
  author={Xue, Chaocan and Zhong, Bineng and Liang, Qihua and Zheng, Yaozong and Li, Ning and Xue, Yuanliang and Song, Shuxiang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  month={June},
  year={2025},
  pages={6730-6740}
}
```

---

# SGLATrack Teknik Dokümantasyon

> **İHA Takibi İçin Benzerlik Rehberli Katman-Uyarlamalı Görüntü Transformeri**
>
> Chaocan Xue, Bineng Zhong, Qihua Liang, Yaozong Zheng, Ning Li, Yuanliang Xue, Shuxiang Song
>
> **CVPR 2025** | [Makale (arXiv)](https://arxiv.org/abs/2503.06625) | [GitHub](https://github.com/GXNU-ZhongLab/SGLATrack) | [Modeller](https://drive.google.com/drive/folders/1sHL7aFVZFwkPy6js48x-EKfoZC7oJc9X?usp=sharing) | [Ham Sonuçlar](https://drive.google.com/drive/folders/1ss-KQqPsfIXeOcl_h3w6Q09dEk07DjUy?usp=sharing)

## Genel Bakış

SGLATrack (**S**imilarity-**G**uided **L**ayer-**A**daptive Tracker), İHA takip senaryoları için tasarlanmış, görüntü transformeri tabanlı tek nesne takip modelidir. Temel yeniliği, çıkarım sırasında hangi transformer katmanlarının çalıştırılacağını dinamik olarak seçen **katman-uyarlamalı mekanizmadır**. Bu mekanizma, ara özellikler arasındaki kosinüs benzerliğine dayanarak gereksiz hesaplamayı azaltırken takip doğruluğunu korur — kaynak kısıtlı İHA platformları için kritik bir gereklilik.

Model, tek akışlı takip paradigması (OSTrack / AVTrack) üzerine inşa edilmiştir. Şablon ve arama bölgesi tokenları birleştirilerek bir görüntü transformeri omurgası üzerinden ortaklaşa işlenir. Hafif bir MLP, derin transformer katmanlarından hangilerinin aktive edilmesi gerektiğini tahmin eder ve çıkarım sırasında erken çıkış imkanı sağlar.

### Model Varyantları

| Varyant | Omurga | Gömme Boyutu | Derinlik | Başlık | Parametre |
|---------|--------|-------------|----------|--------|-----------|
| **SGLATrack-DeiT*** | DeiT-Tiny Distilled | 192 | 12 | 3 | Hafif |
| SGLATrack-ViT | ViT-Base | 768 | 12 | 12 | Standart |
| SGLATrack-EVA | EVA | — | — | — | Gelişmiş |

\* Bu projede kullanılan birincil varyant.

## Dokümantasyon İndeksi

| Doküman | Açıklama |
|---------|----------|
| [Mimari](architecture.md) | Model mimarisi, katman-uyarlamalı mekanizma, tahmin başlıkları, kayıp fonksiyonları |
| [Eğitim](training.md) | Eğitim pipeline'ı, veri setleri, konfigürasyon, dağıtık eğitim |
| [Çıkarım](inference.md) | Takip pipeline'ı, kare kare işleme, değerlendirme |
| [Entegrasyon](integration.md) | Tracker projesiyle entegrasyon, API referansı, pipeline akışı |
| [Açık Kaynak Yüksek Güvenilirlik Toolchain'leri](safety_toolchains.md) | GCC/Clang ve Rust tabanlı sertleştirilmiş build entegrasyonu |
| [Kıyaslamalar](benchmarks.md) | Hava ve genel veri setlerinde performans sonuçları |

## Hızlı Başlangıç

### 1. Yolları Ayarlama

```bash
cd models/SGLATrack
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

### 2. Önceden Eğitilmiş Ağırlıkları İndirme

```bash
# DeiT-Tiny Distilled omurgası
wget https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth \
  -P models/SGLATrack/pretrained_models/

# SGLATrack kontrol noktası (Google Drive'dan)
# Konum: models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar
```

### 3. Takibi Çalıştırma (Tracker projesi içinde)

```python
from python.tracker.sglatrack_wrapper import SGLATrackWrapper

tracker = SGLATrackWrapper()
tracker.init(ilk_kare, başlangıç_kutusu)  # [x, y, w, h] sol-üst format
kutu, güven = tracker.track(sonraki_kare)
```

## Atıf

```bibtex
@inproceedings{sglatrack,
  title={Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking},
  author={Xue, Chaocan and Zhong, Bineng and Liang, Qihua and Zheng, Yaozong and Li, Ning and Xue, Yuanliang and Song, Shuxiang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  month={June},
  year={2025},
  pages={6730-6740}
}
```
