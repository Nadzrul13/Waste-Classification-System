import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="EcoSort AI - Smart Waste Management", layout="wide")

# Custom CSS untuk tampilan elegant
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stExpander { border: none; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); }
    h1 { color: #1e3a8a; font-family: 'Helvetica Neue', sans-serif; }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid;
        margin-bottom: 20px;
    }
    .hero-section {
        background: linear-gradient(to right, #1e3a8a, #3b82f6); 
        padding: 30px; 
        border-radius: 15px; 
        margin-bottom: 30px; 
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI GAUGE CHART ---
def create_gauge_chart(confidence_val):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence_val * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score (%)", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1e3a8a"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 85], 'color': '#fff3cd'},
                {'range': [85, 100], 'color': '#d1e7dd'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model(model_name):
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "Baseline CNN":
        class BaseCNN(nn.Module):
            def __init__(self, num_classes=2):
                super(BaseCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3)
                self.pool1 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.pool2 = nn.MaxPool2d(2, 2)
                self.conv3 = nn.Conv2d(64, 128, 3)
                self.pool3 = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(128 * 26 * 26, 128)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(128, num_classes)
            def forward(self, x):
                x = self.pool1(F.relu(self.conv1(x)))
                x = self.pool2(F.relu(self.conv2(x)))
                x = self.pool3(F.relu(self.conv3(x)))
                x = torch.flatten(x, 1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        model = BaseCNN(num_classes)
        path = "outputs/Base_CNN_best.pth"
    elif model_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
        path = "outputs/VGG16_best.pth"
    else:  # MobileNetV2
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        path = "outputs/MobileNetV2_best.pth"

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# --- SIDEBAR: KONFIGURASI ---
with st.sidebar:
    st.title("EcoSort Control Panel")
    selected_model = st.selectbox("Pilih Model AI", ["MobileNetV2", "VGG16", "Baseline CNN"])
    st.info("MobileNetV2 direkomendasikan untuk kecepatan tinggi.")
    st.divider()
    st.markdown("### Informasi Sistem")
    st.write("🌍 **Tujuan:** Klasifikasi Sampah Pintar")
    st.write("💻 **Runtime:** PyTorch x Streamlit")

# --- MAIN CONTENT ---
st.title("♻️ EcoSort AI: Waste Classification System")
tab1, tab2, tab3 = st.tabs(["🚀 Klasifikasi Real-time", "📊 Analisis Performa", "ℹ️ Tentang Proyek"])

# TAB 1: KLASIFIKASI 
with tab1:
    # 1. Hero Section
    st.markdown("""
        <div class="hero-section">
            <h2 style="margin: 0; color: white;">🚀 Mulai Klasifikasi Sampah Anda</h2>
            <p style="opacity: 0.9; font-size: 1.1em;">Gunakan kekuatan AI untuk mengidentifikasi kategori sampah secara instan dan akurat.</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. Step-by-Step Guide
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("📸 **1. Unggah Gambar**\nPilih foto sampah organik atau anorganik.")
    with s2:
        st.markdown("⚙️ **2. Pilih Model**\nAtur model favorit Anda di sidebar kiri.")
    with s3:
        st.markdown("✅ **3. Lihat Hasil**\nDapatkan kategori dan saran pengolahan.")
    
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Input Gambar")
        uploaded_file = st.file_uploader("Unggah gambar sampah (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Preview Gambar", use_container_width=True)
        else:
            st.info("👋 Menunggu input gambar... Silakan unggah gambar untuk memulai analisis.")

    with col2:
        st.markdown("### 🔍 Hasil Analisis")
        if uploaded_file:
            with st.spinner('AI sedang menganalisis...'):
                model = load_model(selected_model)
                
                # Preprocessing
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0)
                
                # Prediksi
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = F.softmax(output, dim=1)
                    confidence, pred = torch.max(prob, 1)
                
                # Load label mapping
                with open("outputs/label_mapping.json", "r") as f:
                    labels = json.load(f)
                
                res_label = labels[str(pred.item())]
                conf_score = confidence.item()
                
                # Desain Tampilan Hasil (Enhanced Card)
                bg_color = "#d1e7dd" if res_label == "Organic" else "#cfe2ff"
                border_color = "#198754" if res_label == "Organic" else "#0d6efd"
                text_color = "#0f5132" if res_label == "Organic" else "#084298"
                
                st.markdown(f"""
                    <div class="result-card" style="background-color: {bg_color}; border-color: {border_color}; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);">
                        <p style="color:#666; margin:0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8em; font-weight: bold;">Kategori Terdeteksi</p>
                        <h1 style="color: {text_color}; margin: 0; font-size: 2.5em;">{res_label}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                # Visualisasi Confidence (Gauge Chart)
                st.plotly_chart(create_gauge_chart(conf_score), use_container_width=True)
                
                # Expandable Recommendation
                with st.expander("📋 Lihat Rekomendasi Pengolahan", expanded=True):
                    if res_label == "Organic":
                        st.success("**Saran:** Sampah ini dapat diolah menjadi kompos tanaman.")
                    else:
                        st.info("**Saran:** Pisahkan dan serahkan ke fasilitas daur ulang (Bank Sampah).")
        else:
            st.markdown("""
                <div style="text-align: center; padding: 50px; color: #999; border: 1px solid #eee; border-radius: 15px;">
                    <p>Hasil prediksi akan muncul di sini setelah Anda mengunggah gambar.</p>
                </div>
            """, unsafe_allow_html=True)

# --- FUNGSI PLOT TRAINING ---
def plot_training_history(model_name):
    path = f"outputs/history_{model_name}.npy"
    if os.path.exists(path):
        history = np.load(path, allow_pickle=True).item()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # Akurasi
        ax1.plot(history['accuracy'], label='Train Accuracy', marker='o')
        ax1.plot(history['val_accuracy'], label='Val Accuracy', marker='o')
        ax1.set_title(f'Accuracy History: {model_name}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        # Loss
        ax2.plot(history['loss'], label='Train Loss', marker='o', color='red')
        ax2.plot(history['val_loss'], label='Val Loss', marker='o', color='orange')
        ax2.set_title(f'Loss History: {model_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        return fig
    return None

# --- TAB 2: BENCHMARKING ---
with tab2:
    st.header("📈 Model Benchmarking & Training History")
    st.subheader("1. Overall Comparison")
    try:
        st.image("outputs/full_comparison_plot.png", use_container_width=True)
        df = pd.read_csv("outputs/full_model_comparison.csv")
        st.dataframe(df.style.highlight_max(axis=0, subset=['Test Acc (%)']), use_container_width=True)
    except Exception as e:
        st.warning("Data benchmarking belum tersedia.")

    st.divider()
    st.subheader("2. Detailed Training History")
    selected_hist = st.selectbox("Lihat riwayat training untuk model:", ["MobileNetV2", "VGG16", "Base_CNN"])
    with st.spinner(f"Memuat grafik riwayat {selected_hist}..."):
        fig_hist = plot_training_history(selected_hist)
        if fig_hist:
            st.pyplot(fig_hist)
            st.write(f"Grafik di atas menunjukkan proses belajar model **{selected_hist}**.")
        else:
            st.error(f"File history untuk {selected_hist} tidak ditemukan.")

# --- TAB 3: TENTANG WEB ---
with tab3:
    st.markdown("<h2 style='text-align: center; color: #1e3a8a;'>EcoSort AI: Solusi Pemilahan Sampah Pintar</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic;'>Teknologi Deep Learning untuk masa depan bumi yang lebih hijau.</p>", unsafe_allow_html=True)
    st.divider()
    col_about, col_img = st.columns([2, 1])
    with col_about:
        st.markdown("### 🌍 Visi Kami")
        st.write("""
            EcoSort AI hadir untuk membantu memilah sampah secara tepat, mengurangi beban TPA, dan mendukung ekonomi sirkular.
        """)
        st.markdown("---")
        st.markdown("### 🛠️ Keunggulan Teknologi")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("<div style='background-color: #ffffff; padding: 20px; border-radius: 10px; border-top: 5px solid #0d6efd; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h4 style='color: #0d6efd;'>🧠 Multi-Model</h4><p style='color: #555;'>Mendukung 3 arsitektur untuk akurasi optimal.</p></div>", unsafe_allow_html=True)
    with f2:
        st.markdown("<div style='background-color: #ffffff; padding: 20px; border-radius: 10px; border-top: 5px solid #198754; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h4 style='color: #198754;'>🏆 Pre-trained</h4><p style='color: #555;'>Transfer Learning ImageNet standar industri.</p></div>", unsafe_allow_html=True)
    with f3:
        st.markdown("<div style='background-color: #ffffff; padding: 20px; border-radius: 10px; border-top: 5px solid #ffc107; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h4 style='color: #ffc107;'>⚡ Real-time</h4><p style='color: #555;'>Inferensi cepat dengan visualisasi confidence.</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='text-align: center; color: #666;'>Dikembangkan dengan dedikasi oleh <b>Nadzrul Khair</b></p>", unsafe_allow_html=True)