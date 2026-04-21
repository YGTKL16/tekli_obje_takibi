---
description: "Analyze a failing tracker sequence — diagnose why it lost the target, identify failure modes, and propose fixes."
agent: "architect"
argument-hint: "Sequence name, e.g. dataset3/uav1"
---

Verilen dizide tracker'ın neden başarısız olduğunu analiz et.

1. @diagnosis sub-agent'ı kullanarak kare-kare hata analizi yap
2. Hata modlarını sınıflandır (F1-F8)
3. Her hata penceresi için kök neden belirle
4. Spesifik fix öner (parametre veya kod değişikliği)
5. Çapraz doğrulama planı oluştur (aynı kategoriden 5 dizi)

Raporunu yapılandırılmış formatta sun ve Delta tablosu ekle.
