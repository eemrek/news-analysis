# 📰 Haber Metni Sınıflandırma Projesi

<div align="center">
  <img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" width="400" />
  <br>
  <img src="https://www.tensorflow.org/images/tf_logo_social.png" width="400" />
</div>

## 🎯 Proje Hakkında

Bu proje, haber metinlerini içeriklerine göre otomatik olarak sınıflandıran bir derin öğrenme modelini içermektedir. LSTM ve Bidirectional LSTM katmanları kullanılarak geliştirilmiş model, haber başlıklarını ve açıklamalarını analiz ederek kategorize etmektedir.

### 📊 Desteklenen Kategoriler

- 🌍 Dünya Haberleri
- ⚽ Spor Haberleri
- 💼 İş/Ekonomi Haberleri

## 🛠️ Kurulum

```bash
# Gerekli kütüphanelerin kurulumu
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

## 📋 Veri Seti

AG News veri seti kullanılmıştır. Veri seti özellikleri:
- 3 farklı kategori
- Haber başlıkları ve açıklamaları
- Dengeli sınıf dağılımı

## 🚀 Model Mimarisi

```plaintext
Model Katmanları:
├── Embedding Layer (10000, 128)
├── Bidirectional LSTM (64 units)
├── Dropout (0.5)
├── Bidirectional LSTM (32 units)
├── Dropout (0.5)
├── Dense Layer (64 units, ReLU)
└── Output Layer (4 units, Softmax)
```

## 📈 Performans Metrikleri

Model değerlendirme sonuçları:
- Test Doğruluğu: ~92%
- ROC-AUC Skorları
  - Dünya Haberleri: 0.95
  - Spor Haberleri: 0.97
  - İş Haberleri: 0.96

## 📊 Görselleştirmeler

### Confusion Matrix
<div align="center">
  <img src="[https://raw.githubusercontent.com/username/project/master/images/confusion_matrix.png](https://github.com/eemrek/news-analysis/blob/main/confusion-matrix.png)" width="400" />
</div>

### ROC Eğrisi
<div align="center">
  <img src="https://raw.githubusercontent.com/username/project/master/images/roc_curve.png" width="400" />
</div>

## 🔧 Kullanım

```python
# Örnek kullanım
from news_analysis import predict_category

text = "Your news text here"
category = predict_category(text)
print(f"Predicted category: {category}")
```


