# Intel Image Classification — 6 Sınıflı Görüntü Sınıflandırma (CNN ve EfficientNet‑B0)

Bu projede, doğal sahneleri 6 sınıfta sınıflandırdım: buildings (binalar), forest (orman), glacier (buzul), mountain (dağ), sea (deniz), street (sokak). İki yaklaşımı karşılaştırdım:
- Sıfırdan bir CNN (Convolutional Neural Network — fotoğraflardaki kenar/doku gibi desenleri yakalayan katmanlar)
- Transfer learning ile EfficientNet‑B0 (daha önce çok büyük bir veri seti üzerinde öğrenmiş bir modelin bilgisini tekrar kullanıp, en üstteki sınıflandırma katmanlarını bu veri için uyarlama)

Amacım: “Aynı veriyle, sıfırdan eğitim mi yoksa transfer öğrenme mi daha iyi işler çıkarır?” sorusuna net bir cevap vermek; bunu yaparken de “neden böyle yaptım, ne öğrendim?” kısmını açıkça anlatmak.

- Ortam: TensorFlow 2.18, Keras 3, SEED=42 (aynı ayarla tekrar çalıştırılınca benzer sonuçlar için)
- Notebook: `notebooks/intel-image.ipynb`
- Çıktılar: `artifacts/` (grafikler, raporlar, özetler) ve `outputs/` (Grad‑CAM görselleri)

## 1) Veri ve Neden Bu Veri?
- Kaynak: Kaggle “Intel Image Classification”. İçinde 6 sınıf var; bazıları birbirine benziyor (özellikle “glacier” ve “mountain”). Bu benzerlik, model için gerçekçi bir zorluk oluşturuyor.
- Ayrım: Eğitim verisinin %20’sini doğrulama (validation) için ayırdım. Test seti tamamen ayrı bir klasörden geliyor (yani modelin hiç görmediği fotoğraflar).
- Girdi: Fotoğrafları 224×224 boyuta getirdim (çoğu hazır model bu boyutla iyi çalışıyor). Toplu işlem için 32’li paketler halinde modele besledim (batch=32).

Neden fotoğrafları “çeşitlendirme” (data augmentation) yaptım?
- Model, sadece gördüğü örneklere “ezber” yapsın istemiyorum. O yüzden eğitim sırasında fotoğrafları biraz döndürdüm (±15°), hafif kaydırdım/zoomladım (±%10), bazen yatay çevirdim. Bu, gerçek hayattaki farklı açılar/ışıklar gibi küçük değişimleri taklit ediyor ve genellemeyi güçlendiriyor.
- Adil ölçüm için doğrulama ve test kısmında bu oynamaları yapmadım (sadece normalleştirme).

Normalleştirme ne?
- Piksel değerlerini 0–255 aralığından 0–1 aralığına indiriyorum (CNN için). Bu sayısal olarak daha stabil bir eğitim sağlıyor.
- EfficientNet‑B0 tarafında bu iş, modelin içine gömülü “hazırlama” adımıyla (preprocess) yapılıyor. O yüzden burada ayrıca 0–1’e bölmüyorum; iki kere ölçeklememek önemli.

## 2) İki Model: Ne Farkları Var?

### 2.1) Sıfırdan CNN (Basit ve anlaşılır)
- Yapı: 3 bloktan oluşuyor; her blokta “desen yakalayan” konvolüsyon katmanları, ardından “öğrenmeyi dengede tutan” normalizasyon, “özeti alan” havuzlama ve “hafif unutturup ezberlemeyi azaltan” bırakma (dropout) var. Sonda da sınıfları ayıran tam bağlı katmanlar bulunuyor.
- Neden böyle? Bu veri seti için yeterli kapasite (ne çok büyük ne çok küçük) ve anlaşılır bir başlangıç. Dropout ve normalizasyon, erken ezberlemeyi (overfitting) engellemeye yardım ediyor.

### 2.2) EfficientNet‑B0 ile Transfer Learning (Önceden öğrenilmiş bilgiyi tekrar kullanma)
- Fikir: Daha önce çok büyük bir veri seti (ImageNet) üzerinde doğadaki temel desenleri (kenar, doku, renk geçişleri) öğrenmiş bir modelin gövdesini (backbone) alıyorum. En üstteki sınıflandırma katmanlarını (üst katmanlar, classification head) bu projedeki 6 sınıfa göre yeniden eğitiyorum. Böylece başlangıçta sıfırdan “çizgi nedir, doku nedir?” öğretmekle vakit kaybetmiyorum.
- Eğitim planı iki aşama:
  1) Gövdeyi dondur, üst katmanları (classification head) eğit: Önceden öğrenilmiş gövdeye dokunmadan sadece en üstteki sınıflandırma kısmını eğitiyorum. Böylece model hızla temel uyumu yakalıyor.
  2) Gövdenin bir kısmını aç, küçük adımlarla ince ayar yap: Gövdenin son birkaç katmanını (ör. ~20 katman) eğitime tekrar dahil ediyorum. Bu sefer çok küçük bir öğrenme hızıyla ilerliyorum ki önceki bilgi bozulmadan bu veri setine ince dokunuş yapılsın.

Neden EfficientNet‑B0 (VGG16/ResNet yerine)?
- Daha hafif ve hızlı: Sınırlı donanımda çalıştırmak kolay, yine de güçlü sonuç veriyor.
- Dengeli: Derinlik/genişlik/çözünürlük dengesini iyi kurduğu için orta büyüklükteki veri setlerinde çok işe yarıyor.
- Hata riski düşük: Girdi hazırlama (preprocess) adımı modelin içinde olduğu için veri hattı sade ve hataya (ör. iki kere ölçekleme) daha az açık.

Functional API’yı neden kullandım?
- Modeli “parça parça” kurup katmanlara isim vermek mümkün oluyor. Bu sayede Grad‑CAM için “son konvolüsyon katmanını” adıyla bulabiliyorum. Sıralı (Sequential) kurulumda indeksler değişirse kırılma yaşanabiliyor; isim ise sabit kalıyor.

## 3) Eğitim Sırasında Kullandığım “Destekçiler” (ve sade açıklamaları)
- EarlyStopping (erken durdurma): Doğrulama sonucu uzunca bir süre iyileşmezse “buraya kadar” deyip en iyi noktada durduruyor. Gereksiz yere uzun eğitim yapmamış oluyorum.
- ReduceLROnPlateau (öğrenme hızını düşürme): Bir süre iyileşme yoksa “gazı biraz keselim” diyor. Küçük adımlarla daha ince ayar yapılmasına izin veriyor.
- ModelCheckpoint (en iyi modeli kaydet): Eğitim boyunca “en iyi görünen” modeli diske yazıyor; eğitim sonunda geride kalmış bir ağırlıkla kalmıyorum.

Ne yaşadım?
- Bazen 50 epoch’a kadar gitti; çünkü doğrulama kaybındaki minik iyileşmeler “biraz daha dene” dedirtti. En iyi model zaten kaydedildiği için sorun olmadı.
- Öğrenme hızını düşürmek transfer learning tarafında çok işe yaradı; küçük küçük ama kalıcı iyileşmeler geldi.

## 4) Sonuçlar — “Sayılardan çok, ne anlama geliyor?”
- Baseline CNN (test): accuracy ≈ 0.895, loss ≈ 0.332
- EfficientNet‑B0 (test): accuracy ≈ 0.935, loss ≈ 0.188

Accuracy (doğruluk): Tüm test fotoğraflarının yüzde kaçını doğru bildiğim.
Loss (kayıp): Modelin “ne kadar iyi emin olduğunu” da dikkate alan bir ölçü. Düşük loss, sadece doğru sayısının değil, tahminlerin güveninin de yerinde olduğuna işaret eder.

Yorum:
- Transfer learning daha yüksek doğruluk verdi ve özellikle loss tarafında belirgin daha iyi: bu, modelin tahmin “güvenini” de daha iyi ayarladığını gösterir (aşırı emin ama yanlış tahminler daha az).
- En çok karışan sınıflar: glacier ↔ mountain (renk/ton/doku çok benzer). En başarılı sınıflar: forest ve sea (doku ve renk belirgin).

Kanıt dosyaları:
- Confusion matrices (hangi sınıf hangi sınıfla karışmış): `artifacts/baseline_cm.png`, `artifacts/effnet_cm.png`
- Karşılaştırma grafiği (CNN vs TL): `artifacts/compare_cnn_vs_tl.png`
- Sınıf raporları (precision/recall/f1): `artifacts/baseline_report.txt`, `artifacts/effnet_report.txt`
- Özet metrikler: `artifacts/metrics_summary.json`

## 5) Grad‑CAM — “Model nereye bakıyor?”
Grad‑CAM, model karar verirken fotoğrafın hangi bölgelerine dikkat ettiğini gösteren bir ısı haritasıdır.
- Doğru tahminlerde: Orman için yaprak/yeşil doku, deniz için su yüzeyi, binalar için cephe/kenar çizgileri gibi “mantıklı” bölgelere yoğunlaşıyor.
- glacier vs mountain hatalarında: Dikkat alanı daha geniş ve kararsız; net bir odak yok. Bu da bu iki sınıfın görsel olarak ne kadar benzer olduğunu gösteriyor.
- Dosyalar:
  - CNN: `outputs/gradcam_per_class.png`
  - TL: `outputs/gradcam_effnet_per_class.png`

## 6) Sınırlamalar ve Öğrendiklerim
- Veri çeşitliliği: glacier/mountain sahnelerinde ışık/kontrast, çekim koşulları çok değişiyor; bu da ayrımı zorlaştırıyor.
- Şehir sahnelerinde (buildings/street) farklı açı ve ölçekler, bazen benzer dokular oluşturup kafa karıştırabiliyor.
- “Kalibrasyon” (modelin güven puanlarının ne kadar gerçekçi olduğu) nicel olarak ölçülmedi; yorumlardan anlaşılıyor ama ileride ölçüp tabloya koymak iyi olur.
- Hiperparametre aramasını (model mimarisi için geniş tarama gibi) sınırlı tuttum; odak, “CNN vs TL farkını net görmek”ti.

## 7) Geliştirme Önerileri (net ve uygulanabilir plan)

- Veriyi daha anlaşılır hale getirmek
  - En çok karışan fotoğrafları (özellikle glacier–mountain) bir klasöre toplayıp tek tek kontrol edeceğim. Yanlış etiket varsa düzelteceğim.
  - Her sınıftan 20–30 örnekten küçük bir galeri çıkarıp “ayırt edici fark” var mı hızla gözden geçireceğim.

- Fotoğrafları çeşitlendirmek (modelin farklı koşullara alışması için)
  - glacier/mountain: biraz aydınlatıp/karartma (parlaklık/kontrast), azıcık kırpıp yeniden boyutlandırma, hafif bulanıklaştırma.
  - buildings/street: küçük perspektif değişiklikleri (kamera açısı değişmiş gibi) ve hafif gölgeler ekleme.
  - Not: Bunlar gerçek hayattaki ışık ve açı değişimlerini taklit eder; modelin esnekliğini artırır.

- Eğitimi daha “sabırlı ve dengeli” yapmak
  - Eğitim erken durmasın diye bekleme süresini az biraz artıracağım (küçük iyileşmeleri kaçırmamak için).
  - Eğitim adım boyutunu (batch) birkaç değerle deneyeceğim; amaç en sakin ve düzenli ilerleyen eğriyi bulmak.

- “Aşırı emin ama yanlış” tahminleri azaltmak
  - Etiketleri çok az yumuşatan bir ayar (label smoothing) kullanarak modelin gereksiz aşırı güvenini azaltacağım.
  - glacier/mountain sınıflarına eğitimde biraz daha ağırlık verip (class weight), bu iki sınıfın ayrımını güçlendireceğim.

- Benzer fotoğrafları ayırmayı kolaylaştırmak
  - İki fotoğrafı hafifçe karıştıran basit bir teknik (Mixup/CutMix) kısa bir deneme olarak uygulanacak. Amaç, karar sınırını “yumuşatıp” birbirine benzeyen sahneleri daha net ayırmak.

- Transfer öğrenmede küçük ince ayar
  - “Önceden öğrenilmiş” gövdenin ne kadarını açacağımı (ör. son 15/20/25 katman) küçük denemelerle belirleyeceğim. Bu, yeni veriye tam dozunda uyum sağlar.
  - En üstteki sınıflandırma katmanlarında (classification head) bırakma oranını (Dropout) biraz artırmayı deneyeceğim (ör. 0.3 → 0.4); amaç aşırı ezberlemeyi engellemek.

- Açıklanabilirliği (Grad‑CAM) işe koşmak
  - Yanlış tahmin edilen 10 fotoğraf için ısı haritası (Grad‑CAM) galeri çıkarıp “model nereye bakmış?” sorusunu görsel olarak cevaplayacağım.
  - glacier–mountain doğru ve yanlış örnekleri yan yana koyup, odak noktalarının neden farklılaştığını kısaca yorumlayacağım.

- Kullanımı kolaylaştırmak (denemek isteyenler için)
  - Klasördeki fotoğrafları tek seferde okuyup CSV olarak tahmin döken küçük bir komut satırı aracı ekleyeceğim (ör. `python inference.py --input imgs/`).
  - Kurulumu tek satıra indirmek için basit bir ihtiyaç listesi paylaşacağım (`requirements.txt`).
  - README’de tek bakışta görülsün diye: karşılaştırma grafiği, iki karışıklık matrisi ve birkaç Grad‑CAM görselini üstte toplu göstereceğim.

Neden bu plan?
- Hatalar en çok birbirine benzeyen sahnelerde (glacier–mountain). Fotoğrafları çeşitlendirmek ve bu sınıflara biraz daha odaklanmak, hataları somut olarak düşürüyor.
- Eğitimde acele etmemek, sonlara doğru gelen küçük ama kalıcı iyileşmeleri yakalamaya yardım ediyor.
- “Aşırı emin ama yanlış” durumlarını azaltınca model gerçek hayatta daha güvenilir davranıyor.

## 8) Proje Yapısı (Repo)
- `notebooks/intel-image.ipynb`
- `artifacts/`
  - `baseline_cm.png`, `effnet_cm.png`, `compare_cnn_vs_tl.png`, `optimizer_compare.png`
  - `baseline_report.txt`, `effnet_report.txt`, `metrics_summary.json`
  - (model dosyaları) `cnn_best.keras`, `effnet_b0_tl.keras` vb.
- `outputs/`
  - `gradcam_effnet_per_class.png` (TL)
  - `gradcam_per_class.png` (CNN)

## 9) Linkler
- Kaggle not defterim: `https://www.kaggle.com/code/yagmurcorum/intel-image`
