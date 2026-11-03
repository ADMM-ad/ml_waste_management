# Sumber Dataset Klasifikasi Sampah (Real-World Waste Classification Dataset)

Proyek ini menggunakan kurasi dataset citra sampah **nyata (Real-world Garbage Photos)** untuk klasifikasi tiga kategori utama:
1. **Organic Waste** (Sampah Organik: sisa makanan, buah busuk, sayur, dedaunan)
2. **Plastic Waste** (Sampah Plastik: botol minum, gelas plastik, kantong keresek, kemasan plastik makanan)
3. **Paper Waste** (Sampah Kertas: kertas koran, kardus kemasan, majalah, kertas karton)

## Atribusi Sumber Data
1. **Garbage Classification (12 Classes Dataset)**:
   - Sumber: Kaggle Dataset by Mostafa Abla
   - URL: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
   - Lisensi: CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)
   - Deskripsi: Dataset kumpulan foto nyata sampah domestik dan komersial dari berbagai sudut, pencahayaan, dan latar belakang lingkungan nyata.
2. **TrashNet Dataset**:
   - Pengunggah/Penyusun: Gary Thung & Mindy Yang (Stanford University)
   - URL: https://github.com/garythung/trashnet

## Komposisi Dataset Terkurasi
- Total Data: **1.200 Citra Nyata** (Resolusi $224 \times 224$)
  - **Organic Waste**: 400 Citra Asli (280 Train, 60 Validation, 60 Test)
  - **Plastic Waste**: 400 Citra Asli (280 Train, 60 Validation, 60 Test)
  - **Paper Waste**: 400 Citra Asli (280 Train, 60 Validation, 60 Test)
- Proporsi Split:
  - Training Set: 70% (840 citra)
  - Validation Set: 15% (180 citra)
  - Testing Set: 15% (180 citra)
