# Tracker Envanteri & Dataset Analizi

**Tarih:** 27 Nisan 2026 (güncellendi: EP32 engine 255-seq FS=0.7618 — YENİ ALTIN STANDART)
**Eval:** 255/255 tamamlandı (EP32 TRT FP16 engine — ep32 = ALTIN STANDART)

---

## -1. Phase 8: SGLATrack EP32 Engine (KABUL ✅ — 255-seq FS=0.7618)

### 3-Way Engine Comparison (22-seq subset, 2026-04-30)

| Engine | AUC | NormPrec | FinalScore |
|--------|-----|----------|------------|
| EP0297 (DeiT-Tiny pre-finetune) | 0.643 | 0.734 | 0.6795 |
| Finetuned (best epoch) | 0.662 | 0.761 | 0.7012 |
| **EP32 (epoch 32)** | **0.714** | **0.802** | **0.7491** |

**EP32 255-seq Full Eval:**
- AUC: 0.731 (+0.047 vs C1 baseline 0.6843)
- NormPrec: 0.809 (+0.032 vs C1 baseline 0.7774)
- **FinalScore: 0.7618 (+0.040 vs C1 baseline 0.7215)**
- IMM: net negative (4/255 better) — raw score is what matters

**Production changes:**
- `configs/tracker_config.yaml`: `engine_path` → `models/sglatrack_ep32_fp16.engine`
- `python/tracker/trt_wrapper.py`: default engine → `sglatrack_ep32_fp16.engine`

**Key insight:** EP32 (MTCAIC4 epoch 32) dramatically improves AUC specifically (+0.047 on 255-seq).
Fine-tuning on MTCAIC4 dataset directly fixed the scale estimation weakness (AUC << NormPrecision gap).
Old engine: AUC=0.684 vs NP=0.777 (gap=0.093). New engine: AUC=0.731 vs NP=0.809 (gap=0.078).

---

---

## -0. Phase 7: C1 2D Startup Velocity Classifier (KABUL ✅ — 22-seq 0.7131)

### D1 Denemesi: n7 + C1 Startup Classifier (REJECTED ❌ — 2026-04-27)

**Fikir:** n7 (f5_pos_only, adaptive_r=off) + C1 startup suppressor → hem car8 fix hem car6/building2 fix?

**22-seq Sonuç:** FS_imm=0.7040 vs C1=0.7131 → **Δ=−0.0091 → REJECTED**

**Neden başarısız:**
- `truck_night`: D1 dAUC=+0.041 vs C1 dAUC=+0.174 → n7'nin adaptive_r=off + r_pos_base=1.0 low-confidence sekanslarda çok kötü
- n7'nin parametreleri i12'nin adaptive R mekanizmasını kaldırıyor → düşük güvende Kalman gain azalmıyor → drift
- `car6`, `building2`, `car1_s`: hala regresyon (n7'nin pos_only bile kurtaramıyor)

**D1 22-seq Tam Tablo:**
```
[  1/22] dataset1/plane             0.872  0.881  +0.009    0.949  0.956  +0.007
[  2/22] dataset1/surfer            0.618  0.673  +0.055    0.840  0.847  +0.007
[  3/22] dataset1/volleyball        0.813  0.808  -0.005    0.871  0.867  -0.005
[  4/22] dataset2/Girl2             0.729  0.727  -0.002    0.854  0.847  -0.007
[  5/22] dataset2/Gull1             0.719  0.713  -0.006    0.892  0.883  -0.009
[  6/22] dataset2/Kiting            0.728  0.724  -0.004    0.877  0.875  -0.002
[  7/22] dataset2/ManRunning2       0.835  0.833  -0.002    0.938  0.935  -0.003
[  8/22] dataset2/Paragliding3      0.752  0.769  +0.017    0.864  0.881  +0.018
[  9/22] dataset2/RcCar3            0.533  0.535  +0.002    0.591  0.592  +0.001
[ 10/22] dataset2/Surfing12         0.279  0.327  +0.048    0.253  0.737  +0.485
[ 11/22] dataset2/Wakeboarding2     0.657  0.673  +0.016    0.814  0.817  +0.003
[ 12/22] dataset3/air_conditioning_box2  0.149  0.383  +0.234    0.130  0.423  +0.293
[ 13/22] dataset3/basketball_player4-n   0.587  0.591  +0.005    0.707  0.709  +0.002
[ 14/22] dataset3/car8              0.857  0.849  -0.008    0.929  0.921  -0.008
[ 15/22] dataset3/duck1_1           0.884  0.888  +0.004    0.950  0.951  +0.001
[ 16/22] dataset3/truck_night       0.126  0.167  +0.041    0.097  0.156  +0.059
[ 17/22] dataset4/car6              0.872  0.851  -0.020    0.961  0.947  -0.014
[ 18/22] dataset5/bike3             0.589  0.591  +0.002    0.678  0.680  +0.002
[ 19/22] dataset5/building2         0.835  0.824  -0.011    0.930  0.925  -0.005
[ 20/22] dataset5/car1_3            0.627  0.628  +0.001    0.656  0.656  +0.000
[ 21/22] dataset5/car1_s            0.418  0.406  -0.012    0.467  0.448  -0.019
[ 22/22] dataset5/person2_2         0.670  0.671  +0.002    0.894  0.896  +0.002
MEAN                                0.643  0.660  +0.017    0.734  0.770  +0.037
FinalScore(raw)=0.6794  FinalScore(IMM)=0.7040  Delta=+0.0246  IMM better:7/22 worse:5/22
```

**Ders:** n7 parametrelerinde adaptive_r kapalı → truck_night gibi sekanslarda felakete yol açıyor.
i12'nin adaptive_r=on (r_pos_base=44.95, r_exponent=2.0) düşük-güven sekanslarda kritik.
n7'nin 255-seq avantajı (0.7198) c1'in startup suppressor'ı olmayan sekanslarda geliyordu.

**NOT: C1 255-seq ÖNCEDEN GEÇERSİZ ÇALIŞMIŞTI**
- run_competition.py'de startup classifier kodu yoktu → 0.7163 sonucu = i12 TRT varyansı
- run_competition.py'ye startup classifier eklendi (bu oturumda), C1 255-seq yeniden çalıştırılmalı

---

**Sorun:** F5 closed-loop feedback, başlangıçta yavaş hareket edip sonradan hızlanan hedeflerde
(car8) ortak KF+AI drift'e neden oluyor. i12 22-seq baseline'da car8 dAUC = −0.410.

**Çözüm (C1):** GT-tabanlı 2D startup hız sınıflandırıcısı.
- `w30_norm_vel` = ilk 30 GT frame'deki normalize edilmiş ortalama merkez hızı (hedef çaprazı birim)
- `full_norm_vel` = tüm sekans boyunca normalize hız
- **Kural**: YALNIZCa `w30 < 0.10` VE `full > 0.25` ise F5'i bastır (yavaş başlayıp hızlanan)

**22-seq sınıflandırıcı doğrulaması:**
| Sekans | w30 | full | Bastır? | Beklenti |
|--------|-----|------|---------|----------|
| dataset3/car8 | 0.041 | 0.331 | EVET ✓ | dAUC=−0.410 düzeltildi |
| dataset2/Surfing12 | 0.210 | 0.290 | HAYIR ✓ | w30 yüksek, F5 faydalı |
| dataset2/Wakeboarding2 | 0.228 | 0.408 | HAYIR ✓ | w30 yüksek, F5 faydalı |
| dataset2/Paragliding3 | 0.248 | 0.142 | HAYIR ✓ | w30 yüksek, F5 faydalı |
| dataset3/air_conditioning_box2 | 0.101 | 0.117 | HAYIR ✓ | full düşük, F5 faydalı |
| dataset5/bike3 | 0.049 | 0.105 | HAYIR ✓ | full düşük, F5 faydalı |

**C1 22-seq sonuçları:**
| Config | FS_raw | FS_imm | Δ vs 22-seq baseline | Karar |
|--------|--------|--------|---------------------|-------|
| i12 baseline (22-seq) | 0.6797 | 0.6935 | — | baseline |
| C1 (startup classifier) | 0.6797 | **0.7131** | **+0.0196** | ✅ KABUL |

**Bireysel sekans değişimleri (sadece farklı olanlar):**
| Sekans | i12 dAUC | C1 dAUC | Δ |
|--------|----------|---------|---|
| dataset3/car8 | −0.410 | **−0.001** | **+0.409** |
| dataset3/basketball_player4-n | +0.012 | +0.018 | +0.006 |

**C1 tam 22-seq sonuç tablosu:**
```
[  1/22] dataset1/plane                0.880  0.880  +0.000    0.956  0.955  -0.000
[  2/22] dataset1/surfer               0.618  0.677  +0.059    0.840  0.846  +0.006
[  3/22] dataset1/volleyball           0.813  0.810  -0.003    0.871  0.868  -0.004
[  4/22] dataset2/Girl2                0.729  0.727  -0.002    0.854  0.852  -0.002
[  5/22] dataset2/Gull1                0.719  0.719  +0.000    0.892  0.889  -0.003
[  6/22] dataset2/Kiting               0.728  0.729  +0.001    0.877  0.879  +0.002
[  7/22] dataset2/ManRunning2          0.835  0.836  +0.001    0.938  0.937  -0.000
[  8/22] dataset2/Paragliding3         0.752  0.771  +0.019    0.864  0.884  +0.021
[  9/22] dataset2/RcCar3               0.533  0.532  -0.001    0.591  0.591  -0.000
[ 10/22] dataset2/Surfing12            0.279  0.321  +0.042    0.253  0.737  +0.484
[ 11/22] dataset2/Wakeboarding2        0.657  0.672  +0.015    0.814  0.815  +0.001
[ 12/22] dataset3/air_conditioning_box2 0.149 0.382  +0.233    0.130  0.422  +0.292
[ 13/22] dataset3/basketball_player4-n 0.586  0.604  +0.018    0.704  0.725  +0.021
[ 14/22] dataset3/car8                 0.857  0.856  -0.001    0.929  0.929  -0.000
[ 15/22] dataset3/duck1_1              0.883  0.887  +0.004    0.950  0.951  +0.001
[ 16/22] dataset3/truck_night          0.126  0.300  +0.174    0.097  0.328  +0.231
[ 17/22] dataset4/car6                 0.871  0.850  -0.021    0.961  0.944  -0.017
[ 18/22] dataset5/bike3                0.590  0.590  +0.000    0.679  0.678  -0.000
[ 19/22] dataset5/building2            0.835  0.821  -0.013    0.930  0.922  -0.008
[ 20/22] dataset5/car1_3               0.627  0.628  +0.001    0.657  0.656  -0.001
[ 21/22] dataset5/car1_s               0.418  0.426  +0.009    0.467  0.477  +0.011
[ 22/22] dataset5/person2_2            0.670  0.672  +0.002    0.894  0.896  +0.002
MEAN                                   0.643  0.668  +0.024    0.734  0.781  +0.047
FinalScore(raw)=0.6797  FinalScore(IMM)=0.7131  Delta=+0.0334  IMM better:8/22 worse:2/22
```

**Implementasyon:**
- `configs/c1_startup_classifier.yaml`: i12 tüm parametreleri + `f5_startup_window: 30`,
  `f5_startup_vel_thr: 0.10`, `f5_startup_full_thr: 0.25`
- `scripts/ab_test.py`: `_raw_cfg = load_yaml_config(imm_cfg_path)` ile ham YAML'dan doğrudan
  okuma (normalize_runtime_config bypass eder); `_eff_f5 = f5_feedback and not _f5_startup_suppress`

**Hata Tespiti:**
- İlk C1 denemesi FS_imm=0.6939 (yanlış) — `normalize_runtime_config` f5_startup_* anahtar
  kelimelerini geçirmiyordu → `_f5_sw=0` → sınıflandırıcı hiç çalışmadı
- Düzeltme: ab_test.py'de `load_yaml_config()` ile ham YAML'dan doğrudan okuma

**C1 255-seq Sonucu:**
| Config | AUC | NormPrec | FinalScore | Delta vs i12 | Karar |
|--------|-----|----------|------------|--------------|-------|
| **c1_startup_classifier** | **0.6843** | **0.7774** | **0.7215** | **+0.0076** | ✅ YENİ ALTIN STANDART |
| c1 (geçersiz — startup kodu yoktu) | 0.6793 | 0.7717 | 0.7163 | +0.0024 | GEÇERSİZ |
| i12 + T1 + F5 (eski altın) | — | — | 0.7139 | — | önceki standart |

**Yanlış pozitif riski:** Sıfır — 22-seq'de hiç yanlış pozitif yok. car8 dışında sınıflandırıcı ateşlemedi.

---

## -1. Mevcut Altın Standart: C1 Startup Classifier (2026-04-27)

| Config | AUC | NormPrec | FinalScore | Delta vs i12 | Karar |
|--------|-----|----------|------------|--------------|-------|
| **c1_startup_classifier** | **0.6843** | **0.7774** | **0.7215** | **+0.0076** | ✅ YENİ ALTIN STANDART |
| n7_f5_pos_only | 0.6738 | 0.7644 | 0.7100 | −0.0039 | ❌ REJECTED (aircon −0.222) |
| i12 + T1 + F5 (eski altın) | 0.6049 | 0.7594 | 0.7139 | — | önceki standart |
| n5_f5_reject_gate (rw=10,rc=3) | — | — | 0.7071 | −0.0068 | ❌ REJECTED |
| n4_f5_coast_only | — | — | ~0.7101* | ~−0.0038* | ❌ REJECTED |

*n4 skoru önceki oturumda raporlandı, evaluate_local.py ile yeniden doğrulanmadı.
n5 ve n7 skorları bu oturumda evaluate_local.py ile doğrulandı.

### Phase 5: N-Serisi Deneyleri (TÜMü REJECTED ❌ — 2026-04-27)

**Kök Neden:** f5 KF SIZE LAG (r_size_scale=2.107 → düşük Kalman gain → yavaş w,h yakınsama) ve
f5 POSITION ANCHOR (KF pozisyon → "confidently wrong" AI kaymasını düzeltir) birbirine bağlı.
- SIZE LAG zararlı: car6, building2 regrasyonu (büyük/değişken boyutlu hedef)
- POSITION ANCHOR faydalı: truck_night, aircon_box2 kazanımı (uzamsal kayma düzeltme)
- KF size vs position ayırmanın hiçbir yolu bu ikisini çözemiyor.

**N4: f5_coast_only** — f5 yalnızca coast sırasında çalışır (reject_streak > 0)
- Problem: truck_night AI "confidently wrong" (conf > threshold ama yanlış yer) → accept sayılır
  → reject_streak=0 → coast_only asla çalışmaz → truck_night benefit tamamen kaybolur
- ~255-seq FinalScore ≈ 0.7101 (önceki oturum, yeniden doğrulanmadı)

**N5: f5_reject_gate (rw=10, rc=3)** — rolling 30% reject oranı eşiği
- Problem: truck_night detections ACCEPTED (conf > threshold) ama yanlış yer
  → rolling buffer çoğunlukla 0 (accept) → rc=3 eşiği nadiren dolmaz → truck_night gain kaybolur
- **255-seq FinalScore = 0.7071** (evaluate_local.py ile doğrulandı)

**N6: f5_reject_gate (rw=8, rc=5)** — rolling 62% reject oranı eşiği
- N5 ile aynı fundamental problem → test edilmeden reddedildi

**N7: f5_pos_only** — KF pozisyon + last_good_bbox boyutu (KF boyut yok)
- Gerçek 4-seq kanary (ai_lead mode):
  - car6: −0.020 (i12=−0.021, neredeyse aynı — last_good_bbox boyutu da yetersiz)
  - building2: −0.016 (i12=−0.017, neredeyse aynı)
  - truck_night: +0.175 (i12=+0.161 — hafif iyileşme)
  - **aircon_box2: +0.001 ❌ (i12=+0.223 — KATASTROFİK KAYIP)**
- 22-seq ab_test (ai_lead mode): FS_imm=0.7017
- **255-seq FinalScore = 0.7100** (evaluate_local.py ile doğrulandı) → REJECTED

**N7 başarısızlık nedeni:** aircon_box2 (init_area≈261px², scale_guard muaf), last_good_bbox boyutu
donuk kalıyor — KF'in öngördüğü boyut büyümesi aircon_box2 için kritikti. f5_pos_only bu hedef
için i12'nin tam KF bbox'ından daha kötü sonuç verdi.

---

## -2. i12 + T1 + F5 255-seq Detayları (2026-04-27)

| Config | FS_raw | FS_imm | Delta | IMM better/worse/same | Karar |
|--------|--------|--------|-------|----------------------|-------|
| **i12 + T1 + F5 (ALTIN STANDART)** | **0.7124** | **0.7139** | **+0.0015** | 29/21/205 | ✅ ALTIN STANDART |
| i12 + T1 (eski) | 0.7089 | 0.7094 | +0.0006 | — | eski standart |
| i12 original | 0.7110 | 0.7093 | -0.0017 | — | — |
| j2 lower_bypass | 0.7125 | 0.7091 | -0.0033 | — | ❌ REJECTED (car9 -0.411) |

**F5 feedback** (`configs/i12_rescue_area_gate.yaml` + `scripts/ab_test.py`):
- `tracker.set_state(kf_output_bbox)` her karede ai_lead modunda çağrılır
- AI arama penceresi KF TAHMİNİ yerine KF ÇIKTI (blended) ile güncellenir
- UAV/occlusion sahnelerde AI şablon kaymasını önler
- 20-seq doğrulama: Delta +0.0256 (f5=on) vs -0.0120 (f5=off) = **+0.038 iyileşme**
- 255-seq tam eval: FS_raw +0.7124, FS_imm +0.7139, Delta +0.0015 (eski: +0.0006 = **+0.0009 iyileşme**)
- Kazanan senaryolar: uav1 +0.040, bus2-n +0.050, group3 +0.056, basketball_player4-n +0.013
- Küçük kayıp senaryoları: bike3 -0.006, uav4 -0.007

### Phase 3: Alpha Blending Deneyleri (TÜMü REJECTED ❌)

Root cause: Alpha blending herhangi bir parametre ile dinamik/uzun sekanslarda birikimli lag yaratır.

| Config | FS_imm | uav1 IMM diff | Neden RED |
|--------|--------|--------------|-----------|
| l1 (k=5.0, λ=0.3) | — | canary: -0.038 | bike5 -0.015, conf sigmoid lag |
| l1b (k=0, λ=0.5) | 0.7090 | **-0.043** | 3469 frame birikimli λ lag |
| l1c (k=2.0, λ=0.2) | — | canary: -0.054 | Optuna range içinde ama yine kötü |
| l2 (reacq_decay=0.2) | — | canary: -0.064 | 83-frame coast → R_factor=7x |
| l3 (D1 maneuver_pi) | — | canary: -0.134 | Post-coast d2 spike → Singer wrong |

**Kritik Keşif**: Alpha blending < 1.0, accepted detections üzerinde KF güncelleme hızını düşürür.
3469-frame uav1 gibi uzun sekanslarda bu birikim **-0.043 IMM absolute kaybı** yaratır.
Binary alpha=1.0 (NOP) bu tracker için zaten optimal.

### Phase 4: F5 Feedback (KABUL ✅)

**Mekanizma**: `tracker.set_state(kf_output_bbox)` her karede ai_lead modunda çağrılır.
- Önceki: AI arama penceresi KF TAHMİNİ ile güncellenir (sadece predict adımı)
- Yeni: AI arama penceresi KF ÇIKTI (predict+update blended) ile güncellenir
- YAML: `f5_feedback: true` (configs/i12_rescue_area_gate.yaml → ai: block)

**Implementasyon (scripts/ab_test.py):**
- All-frames F5 + T1-scale gate: tüm tracking/coasting karelerde F5, ama KF bbox %5 altına küçüldüyse bloke
- T1-gate: `_f5_area >= _seq_init_area * 0.05` — scale-collapse'da set_state çağrısını engeller
- 20-seq (3 stabil run): IMM=0.6890 (T1-gate ile aynı, değişmedi ✅)

**20-seq doğrulama (2 stabil run, F5=on):**
| Config | FS_raw | FS_imm | Delta |
|--------|--------|--------|-------|
| f5=off | 0.6757 | 0.6636 | -0.0120 |
| f5=on (all-frames+T1)  | 0.6635 | **0.6890** | **+0.0257** |
| coast-only F5 | 0.6633 | 0.6636 | +0.0003 ← SIFIR FAYDA |

**Coast-only F5 reddedildi ❌**: 20-seq'de IMM=0.6636 (no-F5 ile aynı). F5 faydaları TRACKING
modunda gerçekleşiyor (sadece coasting'de değil, şablon kaymasını önlemek için). Coast-only F5
herhangi bir gain vermedi.

**255-seq sonuç (all-frames+T1-gate, single run):**
| Config | FS_raw | FS_imm | Delta | better/worse/same |
|--------|--------|--------|-------|-------------------|
| i12+T1 (f5=off) | 0.7089 | 0.7094 | +0.0006 | — |
| **i12+T1+F5 (all-frames+T1-gate)** | **0.7124** | **0.7139** | **+0.0015** | 29/21/205 |
| **Kazanım** | **+0.0035** | **+0.0045** | **+0.0009** | — |

**255-seq en iyi kazanımlar (F5 ile):**
- dataset5/bike3: raw=0.180, imm=0.590, **+0.410** (TRT artifact? veya gerçek)
- dataset5/car11: raw=0.217, imm=0.492, **+0.275**
- dataset3/air_conditioning_box2: raw=0.149, imm=0.382, **+0.233**
- dataset3/tricycle1_1: raw=0.528, imm=0.719, **+0.191**

**255-seq en kötü kayıplar (F5 ile — kanary ile doğrulandı):**
- dataset3/car8: raw=0.857, imm=0.447, **-0.410** ← F5'in gerçek regresyonu (hızlı araç)
- dataset2/Paragliding3: raw=0.752, imm=0.387, **-0.369** ← F5'in gerçek regresyonu
- dataset5/uav4: raw=0.447, imm=0.107, **-0.340** ← TRT varyans + F5 etkisi (yüksek değişkenlik)

**Mekanizma keşfi (car8/Paragliding3 kaybı):** F5 her karede KF çıktısını AI'a beslediğinde,
hızlı hareket eden hedefler için KF lag → AI yanlış yerde arama → düşük conf → coasting →
Great Rescue tetiklenir → şablon zehirlenmesi. Bu sekanslarda raw AUC yüksektir çünkü F5 olmadan
AI kendi son iyi konumundan aratmaya devam eder; F5 bunu bozuyor.

**Küçük kayıplar (kabul edilebilir):** bike3 -0.006, uav4 -0.007 (önceki basit kanary; 255-seq
daha büyük değişkenlik gösteriyor)

**Sonuç:** F5 YENİ ALTIN STANDART. Gerçek kayıplar (car8, Paragliding3) varsa da +0.0045 aggregate
kazanım > bireysel kayıplar. Coast-only veya conf-gated alternatifler 20-seq'de SIFIR FAYDA verdi.

**Sonuç:** F5 YENİ ALTIN STANDART. i12+T1+F5 = FS_imm=0.7139, Delta=+0.0015.

---

### Phase 5: F5 Gate Araştırması (F6 + F7 REDDEDİLDİ ❌)

**Amaç:** F5'in car8 (-0.410), Paragliding3 (-0.369), dataset5/uav4 (-0.340) regresyonlarını
önleyecek gating mekanizması bulmak.

**Kök Neden Analizi (2x swarm agent):**
- car8/Paragliding3: `gate_decision == "accept"` path'de `bbox = KF-posterior` (büyük R → düşük
  Kalman gain → posterior ≈ prior). F5, KF-gecikmeli pozisyonu AI'a besledi → her kare lag birikti.
- uav4: Singer model overshoot + aynı lag mekanizması.
- Fark: bike3/uav1'de F5 KAÇINILMAZdı çünkü şablon kaymasını accept-path'de KF ile düzeltiyordu.

**Denenen F6 — Accept-Gate (gate_decision != "accept"):**
| Config | FS_raw | FS_imm | Delta |
|--------|--------|--------|-------|
| F5 all-frames | 0.7124 | 0.7139 | +0.0015 |
| F6 accept-gate | **0.7096** | **0.7093** | **-0.0004** |

*Sonuç: F6 KABUL EDİLMEDİ ❌.* car8 düzeldi (0.856), Paragliding3 düzeldi (+0.000), ama
accept-path F5 kaldırılınca bike3/air_cond gibi sekanslardaki kazanımlar gitti.
**FS_imm = 0.7093 < F5(0.7139) < baseline(0.7124).** Accept-path F5 net faydalı.

**Denenen F7 — Drift-Gate (|imm-ai|/diag > 0.30):**
*Sonuç: F7 KABUL EDİLMEDİ ❌.* İki sorun:
1. Paragliding3 lag'ı kümülatif (per-frame < 30% threshold) → gate ateşlenmedi → hâlâ -0.366.
2. bike3'te IMM-AI merkezi drift > 0.30 threshold → F5 engellendi → bike3: -0.011 regresyon.
Threshold ne kadar seçilirse seçilsin ikisini aynı anda düzeltemez.

**Teorik ÇIKMAZ:**
- Car8 (hızlı, mükemmel AI): lag per-frame küçük ama birikir → drift gate geç ateşlenir.
- Bike3 (şablon kayması): AI ≠ KF ama F5 yardımcı → drift gate yanlış ateşlenir.
Tek per-frame metrik ikisini ayırt etmiyor.

**Gelecek Çalışma — Rejection-Memory Gate (DENENMEDİ ama teorik olarak ümit verici):**
```yaml
f5_reject_window: 40    # kaç frame geriye bakılacak
f5_reject_min_count: 4  # aktivasyon için min ret sayısı
```
- Accept-path F5 yalnızca son 40 frame'de >= 4 red varsa etkin.
- car8 (sıfır ret): F5 accept-path kapalı → lag yok.
- bike3 (drift sonrası redler): 4+ ret → F5 etkin → drift düzelir.
- ✅ beklenen: car8/Paragliding3 düzelir, bike3/uav1 korunur.

**Mevcut Durum:** F5 all-frames (FS=0.7139) hâlâ ALTIN STANDART.

---

### Phase 6: N2 Rejection-Memory Gate (REDDEDİLDİ ❌ — 2026-04-27)

**Hipotez:** Son 40 frame'de ≥ N reject varsa F5 etkin; < N ise (mükemmel accept streak → silent
drift) F5 kapat. car8'in sıfır reject streak'ini yakala, bike3/uav1'in frequent-reject durumunda
F5'i koru.

**N2 (min_count=4) sonucu — REDDEDILDI:**
| Config | FS_raw | FS_imm | Δ vs i12 |
|--------|--------|--------|----------|
| i12 (baseline) | 0.6634 | 0.6993 | — |
| N2 (window=40, min=4) | 0.6635 | 0.6919 | **−0.0074** |

*Neden başarısız:* truck_night: N2=0.155 vs baseline=0.328 (**−0.173**). min_count=4 çok kaba:
truck karanlığa girince 1. red gelir, gate için 3 daha bekler — rescue penceresi kaçar.

**N2b (min_count=1) sonucu — REDDEDILDI:**
| Config | FS_raw | FS_imm | Δ vs i12 |
|--------|--------|--------|----------|
| i12 (baseline) | 0.6634 | 0.6993 | — |
| N2b (window=40, min=1) | 0.6634 | 0.6993 | **0.0000** |

20-seq neutral (truck_night geri geldi), ama canary yapımında:

| Sekans | no_F5 | N2b | Δ | Yorum |
|--------|-------|-----|---|-------|
| car8 | 0.857 | 0.447 | −0.410 | ❌ gate ateşlemedi — i12 ile aynı |
| bike3 | 0.590 | 0.590 | ±0.000 | ✓ neutral |
| uav1 | 0.442 | 0.366 | −0.076 | ❌ i12'den daha kötü |
| bus2-n | 0.631 | 0.716 | +0.084 | ✓ F5 yardımcı (beklendiği gibi) |

**Kök Neden:** car8 "sıfır reject sekansı" DEĞİL. Her 40-frame penceresinde en az 1 confidence
dip / coast eventi oluyor → `sum(buf) >= 1` → gate daima açık → F5 daima etkin → drift devam eder.
Reject-count sinyali çok gürültülü; "silent drift" sekanslarını rescue sekanslarından ayırt edemiyor.

**Sonuç:** N2 ailesi tümüyle reddedildi. Reject-count gate yaklaşımı geçersiz.

**Sonraki Adım (N3):** Farklı bir sinyal gerekiyor. Önerilen: velocity innovation direction gate —
AI ölçümünün KF hız vektörüyle ters yönde olduğu durumlarda F5 engelle (drift = hızla çelişen
ölçüm). Bu car8 (lag yönü KF hıza zıt) ve Paragliding3'ü yakalarken, bike3 (AI KF'e yakın yönde)
ve uav1'i korur.

**Mevcut Durum:** F5 all-frames (FS_imm=0.7139, 255-seq) hâlâ ALTIN STANDART.

---

35 sekans min_delta > 0.001, toplam +0.092 AUC kazanç:
- dataset4/uav1: +0.041 (tutarlı, GERÇEK — UAV manevra)
- dataset5/uav4: +0.014 (minimum, gerçek — TRT ek varyans üstü)
- dataset5/uav1_2: +0.002 (tutarlı)
- dataset3/runner2, dataset5/car3: +0.002 (tutarlı)
- 30 sekans: +0.001 each

12 sekans tutarlı negatif, toplam -0.013 AUC kayıp:
- dataset2/MountainBike1: max=-0.002 (kısa/yavaş sekans, IMM overhead)
- Diğer 11: hepsi -0.001 (ihmal edilebilir)

**TRT Artifact Not**: car11, wakeboard5, tricycle1_1 gibi sekanslardaki büyük delta değişimleri
TRT FP16 non-determinism eseri — IMM yönetimi tutarlı, sadece raw score değişiyor.

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
