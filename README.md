# Machine Learning Service - Smart Waste Management System

> **Komponen C: Machine Learning Model & REST API Microservice**  
> **Dibuat oleh: Adam Abdurrahman**  
> **Studi Kasus:** Smart Waste Management System  
> **Teknologi:** Python 3.10+, TensorFlow 2.18, Keras, MobileNetV2, FastAPI, Uvicorn, Scikit-learn, OpenCV, Pillow

---

## 🔗 Ekosistem Repository Terkait & Dataset

Microservice kecerdasan buatan ini merupakan bagian dari ekosistem **Smart Waste Management System** yang terbagi dalam tiga repositori terpisah:

- 🌐 **Website & Backend (Laravel):** [https://github.com/ADMM-ad/laravel_waste_management](https://github.com/ADMM-ad/laravel_waste_management)
- 📱 **Mobile Application (Flutter):** [https://github.com/ADMM-ad/flutter_waste_management](https://github.com/ADMM-ad/flutter_waste_management)
- 🤖 **Machine Learning Service (Repository Ini):** [https://github.com/ADMM-ad/ml_waste_management](https://github.com/ADMM-ad/ml_waste_management)
- 📊 **Sumber Dataset AI (Kaggle):** [https://www.kaggle.com/datasets/mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

---

## 🤖 Deskripsi Proyek

Modul **Machine Learning Waste Management** adalah microservice berbasis kecerdasan buatan (*AI*) yang berfungsi mengenali dan mengklasifikasikan jenis sampah secara otomatis berdasarkan citra foto yang diunggah dari aplikasi Mobile (Flutter) maupun Portal Web (Laravel).

Model dibangun menggunakan teknik **Transfer Learning** dengan arsitektur dasar **MobileNetV2** yang telah dilatih pada bobot *ImageNet*. Arsitektur ini dipilih karena memiliki efisiensi komputasi yang sangat tinggi, ukuran berkas model yang ringan (`~9 MB`), serta latensi inferensi yang sangat rendah (*sub-100ms*) sehingga sangat ideal untuk implementasi aplikasi *real-time*.

---

## 🏷️ Kategori Klasifikasi Sampah (3 Kelas)

Sistem mengklasifikasikan sampah ke dalam 3 kategori utama sesuai standar pengelolaan sampah dan pedoman PRD:
1. **Organic Waste (Sampah Organik):** Sisa makanan, sayuran busuk, kulit buah, sisa daging, dedaunan kering, dan sisa bahan hayati yang dapat diurai/dikompos.
2. **Plastic Waste (Sampah Plastik):** Botol plastik minuman (PET), kantong kresek, kemasan sachet, gelas plastik, dan sedotan.
3. **Paper Waste (Sampah Kertas):** Kardus karton, kertas dokumen/arsip, surat kabar/koran bekas, buku, dan majalah.

---

## 📊 Dataset & Metrik Evaluasi Model

### Sumber Dataset
- **Sumber Data Resmi:** [Kaggle Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) (Mostafa Abla, lisensi CC BY-SA 4.0) dan *TrashNet Dataset* (Stanford University).
- **Jumlah Data:** **1.200 citra nyata** (jauh melampaui standar minimal PRD yaitu 150 gambar).
  - *Organic Waste:* 400 Citra (280 Train, 60 Val, 60 Test)
  - *Plastic Waste:* 400 Citra (280 Train, 60 Val, 60 Test)
  - *Paper Waste:* 400 Citra (280 Train, 60 Val, 60 Test)
- **Pembagian Split Data:** 70% Training (840 gambar), 15% Validasi (180 gambar), dan 15% Testing mandiri (180 gambar).

### Hasil Pengujian & Evaluasi (Test Set)

| Metrik | Target Spesifikasi PRD | Capaian Aktual Model | Status |
| :--- | :--- | :--- | :--- |
| **Akurasi Pengujian (Test Accuracy)** | $\ge 80.00\%$ | **100.00%** | ✓ Terpenuhi Sempurna |
| **Test Loss** | Minimal | **0.0039** | ✓ Optimal |
| **Precision (Rata-rata)** | $\ge 0.80$ | **1.00** | ✓ Sempurna |
| **Recall (Rata-rata)** | $\ge 0.80$ | **1.00** | ✓ Sempurna |
| **F1-Score (Rata-rata)** | $\ge 0.80$ | **1.00** | ✓ Sempurna |

*Berkas artefak evaluasi (kurva pelatihan, confusion matrix, dan classification report) tersimpan di direktori `evaluation/` dan dapat dilihat langsung di dashboard web AI.*

---

## 🌐 Dokumentasi Endpoint REST API (FastAPI)

Microservice berjalan secara independen pada port `8000`:

| Method | Endpoint | Deskripsi | Input / Payload | Output Utama |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Dashboard Evaluasi & Performa AI | - | Antarmuka HTML metrik & grafik evaluasi |
| `GET` | `/health` | Pemeriksaan Kesiapan Model & Status | - | `{"status": "healthy", "model_loaded": true}` |
| `POST` | `/predict` | Klasifikasi Citra Sampah | `multipart/form-data` (`file`) | `{"predicted_class": "plastic", "confidence_score": 0.98, ...}` |
| `GET` | `/docs` | Dokumentasi Interaktif Swagger UI | - | OpenAPI Specification UI |

---

## 🛠️ Persyaratan Sistem

- Python `>= 3.10`
- Pip & Virtualenv
- Berkas bobot model: `models/waste_classifier.keras` (telah tersedia)

---

## 🚀 Panduan Instalasi & Konfigurasi Port Lokal

### ⚠️ Informasi Port Ekosistem Lokal
- **Machine Learning FastAPI (Service Ini):** Port `8000` $\rightarrow$ `http://127.0.0.1:8000`
- **Laravel Backend & Web:** Port `8080` $\rightarrow$ `http://127.0.0.1:8080`
- *Catatan:* Port 8000 digunakan khusus oleh microservice AI agar tidak terjadi bentrok dengan web server Laravel di port 8080.

---

### Langkah Menjalankan:

1. **Masuk ke direktori:**
   ```bash
   cd machinelearning_waste_management
   ```

2. **Buat dan aktifkan virtual environment Python:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Pasang seluruh dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. *(Opsional)* **Latih ulang model dari awal:**
   ```bash
   python train.py
   ```
   *Script akan otomatis melatih arsitektur MobileNetV2, mengevaluasi data uji 180 citra, memperbarui berkas `models/waste_classifier.keras`, serta menghasilkan kurva akurasi dan confusion matrix.*

5. **Jalankan server FastAPI pada Port 8000:**
   ```bash
   uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
   ```

6. **Buka antarmuka peramban:**
   - **Dashboard Evaluasi & Uji Live AI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Pengujian Unit & Endpoint API

Jalankan pengujian unit mandiri untuk memverifikasi model inferensi dan API endpoint:

```bash
# Pengujian unit pipeline inferensi
python test_unit.py

# Pengujian integrasi API
python test_api.py
```

---

*Hak Cipta © 2026 Adam Abdurrahman — Smart Waste Management System.*
