# 💊 DrugClassifier-WebApp

AI-powered drug prescription classifier. Built with PyTorch (Deep Learning), FastAPI (Backend), and modern CSS (Frontend). Features real-time inference and data scaling.

---

# 💊 DrugClassifier Web App

Bu proje, hastaların klinik ve demografik verilerine dayanarak en uygun ilaç türünü (Drug A, B, C, X, Y) tahmin eden, **PyTorch tabanlı bir derin öğrenme web uygulamasıdır**.  

Model, **FastAPI** kullanılarak bir web servisine dönüştürülmüş ve modern, kullanıcı dostu bir arayüz ile sunulmuştur.

---

## 📋 Proje Özellikleri

- 🧠 **Derin Öğrenme Modeli**  
  PyTorch ile geliştirilmiş, 5 giriş özelliğine sahip Çok Katmanlı Yapay Sinir Ağı (ANN)

- 🎯 **Yüksek Başarı Oranı**  
  Test verileri üzerinde **%97.5 doğruluk (accuracy)**

- ⚙️ **Gerçek Zamanlı Veri Ölçeklendirme**  
  Eğitim sırasında oluşturulan `StandardScaler` nesnesi (`scaler.joblib`) ile anlık veri dönüşümü

- 🌙 **Modern ve Responsive UI**  
  Kullanıcı dostu, karanlık mod destekli arayüz

- ⚠️ **Güvenlik Mekanizması**  
  **DrugY (yüksek riskli)** tahminlerinde özel görsel uyarı sistemi

---

## 📊 Model Performansı

Model, Kaggle üzerindeki **Drug Classification** veri seti kullanılarak eğitilmiştir.  
Eğitim süreci `Drug.ipynb` dosyasında detaylı şekilde yer almaktadır.

- **Final Test Accuracy:** %97.50  
- **Final Loss:** 0.0022  

### 📉 Karışıklık Matrisi (Confusion Matrix)

```python
confusion_matrix = [
    [17, 0, 0, 0, 1],
    [0, 5, 0, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 11]
]
```

📌 Model yalnızca **1 adet hatalı tahmin** yapmıştır.


## 🚀 Kurulum ve Çalıştırma

### 1. Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```
### 2. Web Servisini Başlatın
```python
python main.py
```
### 3. Uygulamaya Erişim

Uygulamayı çalıştırdıktan sonra aşağıdaki adresten erişebilirsiniz:

http://localhost:8000

## 📁 Proje Yapısı

```text
DrugClassifier-WebApp/
│
├── main.py                         # FastAPI backend ve model inference
├── Drug.ipynb                      # Model eğitimi ve veri analizi
├── drug_classification_model.pth   # Eğitilmiş model ağırlıkları
├── scaler.joblib                  # StandardScaler nesnesi
├── index.html                     # Frontend arayüz
└── README.md                      # Proje dokümantasyonu
```
## 🔧 Teknik Detaylar

Model, kullanıcıdan alınan verileri sayısal formata dönüştürerek işler.

### 🔢 Girdi Dönüşümleri

**Cinsiyet:**
- Kadın (F) → 0  
- Erkek (M) → 1  

**Kan Basıncı (BP):**
- High → 0  
- Low → 1  
- Normal → 2  

**Kolesterol:**
- High → 0  
- Normal → 1  

---

### 📤 Çıktı Sınıfları

- 0 → **DrugY** ⚠️  
- 1 → drugA  
- 2 → drugB  
- 3 → drugC  
- 4 → drugX  

---

## 📡 Veri Kaynağı

Kullanılan veri seti:  
**Kaggle - Drug Classification Dataset**

---

## ⚠️ Uyarı

Bu proje, bir **ödev/proje çalışması kapsamında geliştirilmiştir**.  
Üretilen tahmin sonuçları **sadece eğitim amaçlıdır** ve gerçek tıbbi kararlar için kullanılmamalıdır.
