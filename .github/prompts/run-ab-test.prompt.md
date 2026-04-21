---
description: "Run A/B test comparing AI-only vs AI+IMM configurations and report detailed results with Delta analysis."
agent: "architect"
---

AI-only ile AI+IMM konfigürasyonlarını karşılaştıran bir A/B test çalıştır ve sonuçları raporla.

1. @guard sub-agent'ı ile build ve testlerin geçtiğini doğrula
2. 20-dizi A/B testini çalıştır: `python3 scripts/ab_test.py --imm --gmc --adaptive-r`
3. @evolution sub-agent'ı ile sonuçları analiz et:
   - Genel FinalScore Delta
   - Kategori-bazlı kırılım (static, person, vehicle, aerial, animal, bike, water)
   - Top-5 kazanç ve Top-5 kayıp dizileri
4. Negatif Delta varsa, en kötü dizilerde @diagnosis ile kök neden analizi başlat
5. Nihai karar: KABUL / RED / GÖZLEM

Delta tablosu ve kategori kırılımı ile yapılandırılmış rapor sun.
