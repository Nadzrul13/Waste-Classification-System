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

Proyek ini menggunakan **Waste Classification Data** yang bersumber dari Kaggle. Dataset ini memiliki total data yang besar untuk memastikan model mengenali berbagai variasi bentuk sampah.

* **Sumber Dataset:** [Kaggle - Waste Classification Data by Sashaank Sekar](https://www.kaggle.com/datasets/techsash/waste-classification-data)
* **Total Dataset:** 25.077 Citra
* **Jumlah Kelas:** 2 Kategori Utama

| Kategori | Karakteristik Visual | Jenis Material | Jumlah Data |
| :--- | :--- | :--- | :--- |
| **Organic (O)** | Material alami, tekstur tidak beraturan, tanda pembusukan. | Sisa makanan, sayuran, buah, daun. | **13.966** |
| **Recyclable (R)** | Bentuk geometris, tekstur halus/keras, reflektif. | Plastik, kertas, logam, kaca, botol. | **11.111** |

### 🛠️ Alur Pra-pemrosesan
* **Pengubahan Ukuran (Resizing):** Standarisasi citra menjadi $224 \times 224$ piksel (RGB).
* **Normalisasi:** Menggunakan skala *mean* `[0.485, 0.456, 0.406]` dan *std* `[0.229, 0.224, 0.225]` sesuai standar ImageNet.
* **Augmentasi Data:** Menerapkan *Random Horizontal Flip* dan *Random Rotation* ($10^\circ$).
* **Pembagian Data:** Dataset dipisahkan secara sistematis ke dalam folder `train` (22.564 data) dan `test` (2.513 data).

---

## ⚙️ Metodologi Penelitian

Sistem ini mengikuti alur *End-to-End Machine Learning Pipeline* yang sistematis untuk memastikan integritas data dan performa model yang optimal.

```mermaid
graph TD
    %% Nodes Definition
    A([Waste Imagery Database]) 
    B[Preprocessing & Augmentation]
    C{Architecture Selection}
    D[Base CNN Scratch]
    E[VGG16 TL]
    F[MobileNetV2 TL]
    G[Training & Validation]
    H[Performance Analysis]
    I([Streamlit Dashboard])

    %% Flow
    A --> B
    B --> C
    C --- D
    C --- E
    C --- F
    D & E & F --> G
    G --> H
    H --> I

    %% Styling for Dark Mode Synchronization
    %% Menggunakan warna-warna yang cerah namun elegan agar kontras dengan background gelap
    style A fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#fff
    style B fill:#1e293b,stroke:#94a3b8,stroke-width:1px,color:#fff
    style C fill:#1e293b,stroke:#fbbf24,stroke-width:2px,color:#fff
    style D fill:#0f172a,stroke:#475569,stroke-width:1px,color:#cbd5e1
    style E fill:#0f172a,stroke:#475569,stroke-width:1px,color:#cbd5e1
    style F fill:#0f172a,stroke:#475569,stroke-width:1px,color:#cbd5e1
    style G fill:#1e293b,stroke:#94a3b8,stroke-width:1px,color:#fff
    style H fill:#1e293b,stroke:#94a3b8,stroke-width:1px,color:#fff
    style I fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff

    %% Link Styling (Garis Penghubung)
    linkStyle default stroke:#64748b,stroke-width:1px
```

## 🧠 Arsitektur Model

1. **Base CNN (Custom)**: Arsitektur vanilla yang dibangun dari awal sebagai baseline performa mentah.
2. **VGG16 (Transfer Learning)**: Model pre-trained mendalam yang unggul dalam ekstraksi fitur visual yang kompleks.
3. **MobileNetV2 (Transfer Learning)**: Arsitektur ringan yang dioptimalkan untuk performa tinggi dengan latensi rendah (Mobile-ready).

## 📊 Hasil Evaluasi & Analisis Perbandingan

Sistem ini diuji menggunakan data independen (*test set*) untuk memastikan objektivitas hasil. Berikut adalah rangkuman performa ketiga model:

### 1. Tabel Perbandingan Performa

| Model | Training Accuracy | Test Accuracy | Analysis Comparison |
| :--- | :---: | :---: | :--- |
| **MobileNetV2** | **94.85%** | **92.72%** | **Top-Performer.** Generalisasi terbaik dengan akurasi tertinggi dan sangat ringan untuk deployment. |
| **VGG16** | 93.12% | 91.44% | **Stable Performance.** Performa sangat kuat dan stabil, namun membutuhkan memori besar (high latency). |
| **Base CNN** | 88.20% | 86.47% | **Baseline.** Hasil solid untuk arsitektur sederhana, namun masih di bawah performa model *Transfer Learning*. |

### 2. Insight Analisis
* **Generalisasi**: MobileNetV2 menunjukkan selisih terkecil antara akurasi training dan testing, yang menandakan model ini paling tahan terhadap *overfitting*.
* **Efisiensi**: Meskipun VGG16 memiliki performa yang kompetitif, MobileNetV2 lebih unggul untuk penggunaan website karena waktu inferensi yang lebih cepat.
* **Kesimpulan**: Berdasarkan data di atas, MobileNetV2 dipilih sebagai model utama dalam sistem **EcoSort AI** karena keseimbangan sempurna antara akurasi dan kecepatan.
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
```

## 📁 Struktur Direktori
```bash
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
```

---

<br />

<div align="center">
  <img src="https://img.shields.io/badge/Copyright-2025-blue?style=flat-square" alt="Copyright">
  
  ### 🏁 EcoSort AI: Final Project
  **Informatics Engineering** **Universitas Muhammadiyah Malang**
  
  ---
  
  *“Technology for a Greener Future”* **Crafted by: [Nadzrul Khair](https://github.com/Nadzrul13)**

</div>
