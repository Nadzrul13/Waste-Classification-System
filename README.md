<div align="center">

# ♻️ ECOSORT AI: INTELLIGENT WASTE CLASSIFIER
### *Deep Learning Framework: Base CNN vs. VGG16 vs. MobileNetV2*

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge&logo=github)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-Waste_Classification-00D4FF?style=for-the-badge&logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)

**[Dataset Link](https://www.kaggle.com/datasets/techsash/waste-classification-data)** • **[Model Benchmarks](#-hasil-evaluasi--analisis-perbandingan)** • **[Installation](#-panduan-instalasi-lokal)**

<br>

</div>

## 📖 Deskripsi Proyek

**EcoSort AI** adalah sistem klasifikasi citra berbasis *Deep Learning* yang dirancang untuk mengidentifikasi kategori sampah secara otomatis. Proyek ini bertujuan untuk mendukung gerakan *smart environment* dengan mempermudah proses pemilahan sampah di sumbernya melalui integrasi teknologi visi komputer.

Sistem ini melakukan studi komparatif antara tiga arsitektur saraf tiruan (**Custom Base CNN**, **VGG16**, dan **MobileNetV2**) untuk menemukan keseimbangan optimal antara akurasi prediksi dan efisiensi komputasi. Aplikasi ini diimplementasikan menggunakan antarmuka web interaktif berbasis Streamlit.

---

## 📂 Dataset & Alur Pra-pemrosesan

Proyek ini menggunakan **Waste Classification Data** yang bersumber dari Kaggle. Dataset ini dirancang untuk melatih model dalam membedakan material sisa berdasarkan potensi daur ulangnya.

* **Sumber Dataset:** [Kaggle - Waste Classification Data by Sashaank Sekar](https://www.kaggle.com/datasets/techsash/waste-classification-data)
* **Jumlah Kelas:** 2 Kategori Utama

| Kategori | Karakteristik Visual | Jenis Material |
| :--- | :--- | :--- |
| **Organic (O)** | Material alami, tekstur tidak beraturan, tanda pembusukan. | Sisa makanan, sayuran, buah-buahan, daun kering. |
| **Recyclable (R)** | Bentuk geometris buatan, tekstur halus/keras, reflektif. | Plastik, kertas, logam, kaca, botol minuman. |

### 🛠️ Alur Pra-pemrosesan
Untuk memastikan performa model yang optimal dan stabil, data diproses melalui tahapan berikut:

* **Pengubahan Ukuran (Resizing):** Standarisasi citra menjadi $224 \times 224$ piksel (RGB) untuk menyesuaikan input arsitektur VGG16 dan MobileNetV2.
* **Normalisasi:** Menggunakan skala *mean* `[0.485, 0.456, 0.406]` dan *std* `[0.229, 0.224, 0.225]` sesuai standar ImageNet.
* **Augmentasi Data:** Menerapkan *Random Horizontal Flip* dan *Random Rotation* ($10^\circ$) pada data latih untuk meningkatkan kemampuan generalisasi model terhadap posisi objek yang bervariasi.
* **Pembagian Data:** Dataset dipisahkan secara sistematis ke dalam folder `train` (latih) dan `test` (uji).

---

## 🧠 Arsitektur Model

1. **Base CNN (Custom)**: Arsitektur vanilla yang dibangun dari awal sebagai baseline performa mentah.
2. **VGG16 (Transfer Learning)**: Model pre-trained mendalam yang unggul dalam ekstraksi fitur visual yang kompleks.
3. **MobileNetV2 (Transfer Learning)**: Arsitektur ringan yang dioptimalkan untuk performa tinggi dengan latensi rendah (Mobile-ready).

## 📊 Hasil Evaluasi & Analisis Perbandingan

### 1. Tabel Perbandingan Performa

| Arsitektur Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **MobileNetV2** | **92.72%** | **92.80%** | **92.70%** | **92.75%** |
| **VGG16** | 91.44% | 91.50% | 91.40% | 91.45% |
| **Base CNN** | 86.47% | 86.50% | 86.40% | 86.45% |

### 2. Insight Analisis
* **MobileNetV2** adalah *Top-Performer* dengan akurasi tertinggi (**92.72%**) dan ukuran model yang paling efisien.
* **VGG16** memberikan performa stabil namun membutuhkan sumber daya memori dan waktu inferensi yang lebih besar.
* **Base CNN** menunjukkan hasil yang solid sebagai arsitektur dasar, namun tertinggal dalam kemampuan generalisasi dibanding model *Transfer Learning*.

---

## 💻 Panduan Instalasi Lokal

```bash
# 1. Clone repositori
git clone https://github.com/Nadzrul13/Waste-Classification-System.git

# 2. Masuk ke direktori proyek
cd Waste-Classification-System

# 3. Instal dependensi library
pip install -r requirements.txt

# 4. Jalankan aplikasi secara lokal
streamlit run app.py

## 📁 Struktur Direktori
Waste-Classification-System/
├── outputs/                # Bobot model (.pth), Log history, & Visualisasi
│   ├── Base_CNN_best.pth
│   ├── MobileNetV2_best.pth
│   ├── VGG16_best.pth
│   ├── full_comparison_plot.png
│   ├── full_model_comparison.csv
│   ├── history_Base_CNN.npy
│   ├── history_MobileNetV2.npy
│   ├── history_VGG16.npy
│   └── label_mapping.json
├── app.py                  # Script Dashboard Streamlit
├── main.ipynb              # Notebook Eksperimen (Training & Eval)
├── requirements.txt        # Dependensi Python
├── structure.txt           # Hierarki file
└── README.md               # Dokumentasi utama
