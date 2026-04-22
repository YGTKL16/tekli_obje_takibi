# Tracker Envanteri & Dataset Analizi

**Tarih:** 16 Nisan 2026 (güncellendi: H3 Singer UAV sonrası)  
**Eval:** 147/255 sekans tamamlandı (GMC + Adaptive-R + baseline mode)

---

## 0. H3 Singer UAV Kaizen Güncellemesi (commit 631c7c0)

**imm_tuned.yaml değişikliği:** `singer.alpha: 1.0 → 3.0`, `singer.sigma2_a: 25.0 → 100.0`

### 20-Sekans Alt-Küme Sonucu
| Config | FinalScore(IMM) | Delta | Karar |
|--------|----------------|-------|-------|
| alpha=1.0 (baseline) | **0.7140** | +0.0511 | ✅ Reference |
| H3 alpha=3.0 | **0.7140** | +0.0505 | ✅ ACCEPTED (aynı skor) |
| H1 pi_persist=0.88 | 0.6999 | +0.0243 | ❌ REJECTED |
| G1A no-refresh | 0.6808 | +0.0173 | ❌ REJECTED |
| G1B conf_high=0.50 | 0.6790 | +0.0157 | ❌ REJECTED |

### H3 Felaket Sekanslara Etkisi — GEÇERLİ alpha=1.0 (güncel kod) KARŞILAŞTIRMASI
**ÖNEMLİ:** Aşağıdaki tablo güncel kod (Faz D + Mahalanobis gate dahil) ile geçerli karşılaştırma.
| Sekans | alpha=1.0 AUC_imm | H3 AUC_imm | H3 kazanımı | Açıklama |
|--------|-------------------|-----------|-------------|-----------|
| dataset3/uav3_1 | **0.370** (-0.406!) | 0.773 | **+0.403!** | Mahal gate kurtarıldı |
| dataset4/uav1 | 0.277 (-0.209!) | 0.434 | **+0.157!** | Mahal gate kurtarıldı |
| dataset3/bus2-n | 0.180 (-0.442!) | 0.623 | **+0.443!** | Occlusion kurtarıldı |
| dataset3/group4_2 | 0.264 (-0.261!) | 0.526 | **+0.262!** | Grup sahne kurtarıldı |
| dataset3/bike5 | 0.221 (-0.155) | 0.375 | **+0.154!** | Bisiklet kurtarıldı |
| dataset5/person1_s | 0.262 (-0.344!) | 0.445 | **+0.183!** | Kişi kurtarıldı |
| dataset4/person7 | 0.319 (-0.465!) | 0.485 | **+0.166!** | Kişi kurtarıldı |
| dataset3/uav1 | 0.626 (-0.022) | 0.646 | +0.020 | UAV iyileştirme |
| dataset5/uav6 | 0.505 (+0.109) | 0.399 | -0.106 | Yavaş UAV regresyon |
| dataset5/uav8 | 0.237 (+0.103) | 0.143 | -0.094 | Yavaş UAV regresyon |
| dataset2/Animal1 | 0.779 (+0.063) | 0.716 | -0.063 | Hayvan regresyon |
| dataset3/runner2 | 0.779 (+0.049) | 0.727 | -0.052 | Koşucu regresyon |
| dataset2/RaceCar | 0.862 (+0.005) | 0.815 | -0.047 | Yarış arabası |
| dataset5/car7 | 0.212 (+0.021) | 0.192 | -0.020 | Araç regresyon |

**Net H3 kazanımı (14 sekans):** +1.788 - 0.382 = **+1.406 AUC**
→ 255 sekansa bölününce ≈ +0.0055 mean AUC, tahmini FinalScore: **+0.007 iyileşme**

### DEFINITIVE 255-Seq Full Train Eval Karşılaştırması
| Config | FinalScore(IMM) | FinalScore(raw) | Delta | IMM better/worse/same |
|--------|----------------|----------------|-------|-----------------------|
| **alpha=1.0 (eski)** | **0.6550** | 0.7102 | **-0.0552** | 52/109/94 |
| **H3 alpha=3.0 (şimdiki)** | **0.7055** | 0.7106 | **-0.0052** | 6/16/233 |
| **H3 kazanımı** | **+0.0505!** | ≈ 0 | +0.0500 | — |

**SONUÇ: H3 alpha=3.0, alpha=1.0'dan 255 sekans üzerinde FinalScore +0.0505 daha iyi!**

20-sekans alt-kümesi neden yanıltıcıydı: 20-seq kurtarma senaryoları için seçilmişti.
Mahalanobis gate felaketi 20-seq'de YOK. Gerçek kompetisyonda H3 dramatik şekilde üstün.



### Mahalanobis Gate Keşfi (Kritik)
alpha=1.0, sigma2_a=25: İnovasyon kovaryansı küçük → χ² skoru büyük → hızlı manevrada AI ölçümleri REDDEDİLİYOR → UAV sekanslarda katastrofik kayblar.
alpha=3.0, sigma2_a=100: Daha büyük kovaryans → daha geniş kapı → hızlı manevrada AI ölçümleri KABUL EDİLİYOR → UAV sekanslarda kurtarma.





---

## 1. Sahip Olduğumuz Enstrümanlar (Bileşenler)

### AI Inference Backend
| Bileşen | Açıklama | Durum |
|---------|----------|-------|
| **SGLATrack DeiT-tiny** | Siamese tracker, ~5.7M parametre | ✅ Aktif |
| **TensorRT FP16 engine** | Hardware-hızlandırılmış çıkarım, 3-5× PyTorch'tan hızlı | ✅ Aktif |
| ONNX export pipeline | PyTorch → ONNX → TRT dönüşümü | ✅ Hazır |

### Kalman Filtre Katmanı (C++ / pybind11)
| Bileşen | Açıklama | Durum |
|---------|----------|-------|
| **KalmanFilter** | 10D state (x,y,w,h + hız + ivme), CV modeli | ✅ Hazır |
| **IMMFilter** | 3 model (CV/CA/Singer) karışımı, otomatik model seçimi | ✅ Aktif |
| **TrackerState** | 3 durum (TRACKING → COASTING → LOST) yaşam döngüsü | ✅ Aktif |
| **Association** | IoU tabanlı top-K aday eşleştirme | ✅ Hazır |
| **Adaptive-R** | Güven skoruna göre ölçüm gürültüsü ayarlama | ✅ Aktif |

### Global Motion Compensation (GMC)
| Bileşen | Açıklama | Durum |
|---------|----------|-------|
| **GMCEstimator** | ORB(500) + RANSAC homografi, downsample=0.5 | ✅ Aktif |
| Foreground masking | Hedef bölgesini GMC'den hariç tutar | ✅ Aktif |
| Failure detection | Homografi bulunamazsa Q artırımı | ✅ Aktif |

### Karar Mekanizması
| Bileşen | Açıklama | Durum |
|---------|----------|-------|
| **DecisionMaker** | Asimetrik güven: update vs coast kararı | ✅ Aktif |
| **IMMPolicy** | step_ai_lead_imm(): AI-liderliğinde hibrit karar mantığı | ✅ Aktif |
| Geometric sanity check | Bbox boyut/pozisyon sıçrama kontrolü | ✅ Aktif |
| Coast/reinit logic | max_coast_frames=30, reinit_after=8 | ✅ Aktif |

### Altyapı & Araçlar
| Bileşen | Açıklama | Durum |
|---------|----------|-------|
| ab_test.py | 255 sekans A/B karşılaştırma | ✅ Aktif |
| replay.py | Önbellek AI çıktıları üzerinde offline test | ✅ Hazır |
| tune_unified.py | Optuna Bayes hiperparametre optimizasyonu | ✅ Hazır |
| Visualizer | OpenCV çizim: bbox, güven, durum renkleri | ✅ Hazır |
| Rust bbox clamping | Yüksek güvenilirlikli bbox doğrulama | ✅ Hazır |

---

## 2. Dataset Bazında Performans (147/255 tamamlandı)

### Per-Dataset Özeti

| Dataset | N | AUC_raw | AUC_imm | dAUC | NP_raw | NP_imm | dNP |
|---------|---|---------|---------|------|--------|--------|-----|
| dataset1 | 9 | 0.754 | 0.751 | -0.003 | 0.845 | 0.838 | -0.007 |
| dataset2 | 53 | 0.668 | 0.664 | -0.005 | 0.773 | 0.769 | -0.004 |
| dataset3 | 84 | 0.686 | 0.678 | -0.009 | 0.760 | 0.751 | -0.009 |
| dataset4 | 1 | 0.818 | 0.817 | -0.001 | 0.921 | 0.921 | +0.000 |

### Nesne Kategorisine Göre Performans

| Kategori | N | AUC_raw | AUC_imm | dAUC | Yorum |
|----------|---|---------|---------|------|-------|
| **static/structure** | 8 | 0.730 | **0.758** | **+0.028** | ✅ EN İYİ — IMM çok faydalı |
| **person/sport** | 49 | 0.680 | 0.680 | +0.000 | ⚪ Nötr |
| **other** | 2 | 0.444 | 0.442 | -0.002 | ⚪ Nötr |
| **bike/moto** | 20 | 0.658 | 0.654 | -0.003 | ⚪ Nötr |
| **water/boat** | 5 | 0.743 | 0.733 | -0.009 | ⚠️ Hafif kayıp |
| **vehicle** | 38 | 0.748 | 0.736 | **-0.012** | 🔴 KAYIP |
| **animal** | 15 | 0.548 | 0.532 | **-0.015** | 🔴 KAYIP |
| **aerial** | 11 | 0.654 | 0.617 | **-0.037** | 🔴 EN KÖTÜ |

---

## 3. En İyi Kazanımlar (IMM Faydalı) ✅

| # | Sekans | AUC_raw | AUC_imm | Kazanç | Neden |
|---|--------|---------|---------|--------|-------|
| 1 | dataset3/air_conditioning_box2 | 0.149 | 0.382 | **+0.233** | Sabit hedef, AI kaybolunca KF kurtarıyor |
| 2 | dataset3/group1 | 0.237 | 0.464 | **+0.227** | Grup sahnesi, coast iyi çalışıyor |
| 3 | dataset3/football_player1_1 | 0.651 | 0.798 | **+0.147** | Hızlı hareket, GMC + IMM büyük kazanç |
| 4 | dataset2/StreetBasketball1 | 0.170 | 0.306 | **+0.136** | Tıkanıklık sahnesi, KF bridge |
| 5 | dataset3/bike4_1 | 0.636 | 0.723 | **+0.086** | Bisiklet takibi, motion model uyumlu |
| 6 | dataset2/MountainBike1 | 0.491 | 0.575 | **+0.084** | Hızlı hareket, IMM tahmini iyi |
| 7 | dataset3/basketball_player1_2-n | 0.437 | 0.494 | **+0.057** | Gürültülü sahne, filtre stabilize |
| 8 | dataset3/duck1_2 | 0.284 | 0.317 | **+0.033** | Küçük hedef, coast yardımcı |
| 9 | dataset2/Paragliding3 | 0.749 | 0.770 | **+0.021** | Hava aracı, düzgün hareket |
| 10 | dataset2/Car2 | 0.179 | 0.195 | **+0.016** | Araç, küçük iyileşme |

---

## 4. Felaket Kayıpları (IMM Zararlı) 🔴

| # | Sekans | AUC_raw | AUC_imm | Kayıp | Kategori | Olası Neden |
|---|--------|---------|---------|-------|----------|-------------|
| 1 | dataset3/uav1 | 0.646 | 0.354 | **-0.292** | aerial | Ani manevra, KF yanlış yöne coast |
| 2 | dataset3/bus2-n | 0.622 | 0.338 | **-0.284** | vehicle | Tıkanıklık/ID switch, KF yanlış bbox |
| 3 | dataset3/runner2 | 0.730 | 0.481 | **-0.250** | person | Koşucu kaybolma, KF uzun coast hatası |
| 4 | dataset2/Animal1 | 0.717 | 0.555 | **-0.162** | animal | Hayvan ani yön değişikliği |
| 5 | dataset3/bike5 | 0.376 | 0.233 | **-0.143** | bike | Bisiklet, zaten düşük AUC daha da kötü |
| 6 | dataset2/RaceCar | 0.857 | 0.734 | **-0.123** | vehicle | Yüksek hız, KF gecikme |
| 7 | dataset3/uav3_1 | 0.777 | 0.654 | **-0.123** | aerial | UAV manevra, KF sapma |
| 8 | dataset3/group4_2 | 0.526 | 0.442 | **-0.084** | person | Grup sahne, yanlış coast |
| 9 | dataset2/Animal4 | 0.287 | 0.225 | **-0.062** | animal | Hayvan düşük AUC, daha da kötü |
| 10 | dataset3/island | 0.892 | 0.837 | **-0.055** | water | Yüksek AUC'yi düşürdü |

---

## 5. Berbat Sekanslar (AUC_raw < 0.3) — AI Zaten Başarısız

| Sekans | AUC_raw | AUC_imm | IMM Etkisi |
|--------|---------|---------|------------|
| dataset3/truck_night | 0.126 | 0.131 | ⚪ Fark yok |
| dataset1/motorcycle | 0.143 | 0.145 | ⚪ Fark yok |
| dataset3/air_conditioning_box2 | 0.149 | **0.382** | ✅ **KURTARDI** |
| dataset3/bike4_2 | 0.154 | 0.154 | ⚪ Fark yok |
| dataset2/StreetBasketball1 | 0.170 | **0.306** | ✅ **KURTARDI** |
| dataset2/Car2 | 0.179 | 0.195 | ✅ Yardım etti |
| dataset2/Gull2 | 0.183 | 0.184 | ⚪ Fark yok |
| dataset3/uav4 | 0.221 | 0.221 | ⚪ Fark yok |
| dataset3/group1 | 0.237 | **0.464** | ✅ **KURTARDI** |
| dataset3/basketball_player2 | 0.270 | 0.270 | ⚪ Fark yok |
| dataset3/duck1_2 | 0.284 | 0.317 | ✅ Yardım etti |
| dataset3/swan | 0.284 | 0.285 | ⚪ Fark yok |
| dataset2/Animal4 | 0.287 | 0.225 | 🔴 **DAHA KÖTÜ** |
| dataset3/hiker1 | 0.295 | 0.298 | ⚪ Fark yok |

---

## 6. Genel Durum (147/255 tamamlandı)

```
AUC_raw  = 0.6848    AUC_imm  = 0.6780    dAUC  = -0.0068
NP_raw   = 0.7707    NP_imm   = 0.7639    dNP   = -0.0068
Score_raw= 0.7192    Score_imm= 0.7123    dScore= -0.0068

IMM better: 17/147 (%12)
IMM worse:  41/147 (%28)
IMM same:   89/147 (%60)
```

### Değerlendirme

| Metrik | Sonuç |
|--------|-------|
| Genel etki | 🔴 **Negatif** (-0.0068 FinalScore) |
| En büyük kazanç | air_conditioning_box2 (+0.233) |
| En büyük kayıp | uav1 (-0.292) |
| İyi kategoriler | static/structure (+0.028), person/sport (nötr) |
| Kötü kategoriler | aerial (-0.037), animal (-0.015), vehicle (-0.012) |
| Felaketler | 10 sekans >0.05 AUC kaybı — bunlar ortalamayı çok çekiyor |

### Temel Sorun
10 felaket sekansı tek başına toplam ~1.5 AUC puanı kaybettiriyor.
IMM kazanımları (top-10 = ~1.0 AUC kazanç) bu kayıpları karşılayamıyor.

**Öncelik:** Bu 10 felaket sekansındaki coast/predict davranışını analiz edip, 
"ne zaman coast'u kesip AI'ya dönmeli" kararını iyileştirmek gerekiyor.
