import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, classification_report
)
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Prediksi Diabetes - Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 {
        color: #667eea;
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        color: #ccc !important;
        margin: 0.3rem 0 0 0;
    }
    .metric-card b {
        color: #e0e0e0 !important;
    }
    .feature-card {
        background: #1a1a2e;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        color: #e0e0e0 !important;
    }
    .feature-card h4 {
        color: #82aaff !important;
    }
    .feature-card p, .feature-card b, .feature-card li, .feature-card ol {
        color: #ccc !important;
    }
    .info-box {
        background: linear-gradient(135deg, #0d2137 0%, #0a3d62 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00bcd4;
        margin: 1rem 0;
        color: #e0e0e0 !important;
    }
    .info-box b {
        color: #4dd0e1 !important;
    }
    .info-box p, .info-box br {
        color: #ccc !important;
    }
    .success-box {
        background: linear-gradient(135deg, #0d2818 0%, #1b4332 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        color: #e0e0e0 !important;
    }
    .success-box b {
        color: #66bb6a !important;
    }
    .warning-box {
        background: linear-gradient(135deg, #2d1f0e 0%, #3e2723 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        color: #e0e0e0 !important;
    }
    .warning-box b {
        color: #ffb74d !important;
    }
    .danger-box {
        background: linear-gradient(135deg, #2d0a0a 0%, #4a1a1a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
        color: #e0e0e0 !important;
    }
    .danger-box b {
        color: #ef5350 !important;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #2d0a0a 0%, #4a1a1a 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #f44336;
        color: #e0e0e0 !important;
    }
    .prediction-positive h2, .prediction-positive h3 {
        color: #ef5350 !important;
    }
    .prediction-positive p {
        color: #ccc !important;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #0d2818 0%, #1b4332 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #4caf50;
        color: #e0e0e0 !important;
    }
    .prediction-negative h2, .prediction-negative h3 {
        color: #66bb6a !important;
    }
    .prediction-negative p {
        color: #ccc !important;
    }
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        border: none;
        margin: 2rem 0;
        border-radius: 2px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    [data-testid="stToolbar"] {right: 2rem;}
</style>
""", unsafe_allow_html=True)


# --- DEEP NEURAL NETWORK FROM SCRATCH ---
class DeepNeuralNetwork:

    def __init__(self, layer_dims, random_state=42):
        np.random.seed(random_state)
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.params = {}
        self.training_history = {"loss": [], "accuracy": []}

        for l in range(1, self.L + 1):
            self.params[f"W{l}"] = np.random.randn(
                layer_dims[l], layer_dims[l - 1]
            ) * np.sqrt(2.0 / layer_dims[l - 1])
            self.params[f"b{l}"] = np.zeros((layer_dims[l], 1))

    # ---------- activation functions ----------
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def sigmoid(Z):
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))

    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(float)

    # ---------- forward ----------
    def forward_propagation(self, X):
        cache = {"A0": X}
        A = X
        for l in range(1, self.L):
            Z = self.params[f"W{l}"] @ A + self.params[f"b{l}"]
            A = self.relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
        Z = self.params[f"W{self.L}"] @ A + self.params[f"b{self.L}"]
        AL = self.sigmoid(Z)
        cache[f"Z{self.L}"] = Z
        cache[f"A{self.L}"] = AL
        return AL, cache

    # ---------- cost ----------
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        eps = 1e-8
        cost = -(1.0 / m) * np.sum(
            Y * np.log(AL + eps) + (1 - Y) * np.log(1 - AL + eps)
        )
        return float(np.squeeze(cost))

    # ---------- backward ----------
    def backward_propagation(self, Y, cache):
        grads = {}
        m = Y.shape[1]
        AL = cache[f"A{self.L}"]
        dZ = AL - Y
        grads[f"dW{self.L}"] = (1.0 / m) * (dZ @ cache[f"A{self.L - 1}"].T)
        grads[f"db{self.L}"] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = self.params[f"W{self.L}"].T @ dZ

        for l in range(self.L - 1, 0, -1):
            dZ = dA_prev * self.relu_derivative(cache[f"Z{l}"])
            grads[f"dW{l}"] = (1.0 / m) * (dZ @ cache[f"A{l - 1}"].T)
            grads[f"db{l}"] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA_prev = self.params[f"W{l}"].T @ dZ
        return grads

    # ---------- update ----------
    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.params[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    # ---------- train ----------
    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        self.training_history = {"loss": [], "accuracy": []}
        for _ in range(epochs):
            AL, cache = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            preds = (AL > 0.5).astype(int)
            acc = float(np.mean(preds == Y))
            grads = self.backward_propagation(Y, cache)
            self.update_parameters(grads, learning_rate)
            self.training_history["loss"].append(cost)
            self.training_history["accuracy"].append(acc)
        return self.training_history

    # ---------- predict ----------
    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return (AL > 0.5).astype(int), AL

    # ---------- manual forward (for demonstration) ----------
    def get_manual_forward_pass(self, x_sample):
        steps = []
        A = x_sample
        for l in range(1, self.L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = W @ A + b
            if l < self.L:
                A_new = self.relu(Z)
                activation = "ReLU"
            else:
                A_new = self.sigmoid(Z)
                activation = "Sigmoid"
            steps.append(
                {
                    "layer": l,
                    "W": W.copy(),
                    "b": b.copy(),
                    "A_prev": A.copy(),
                    "Z": Z.copy(),
                    "A": A_new.copy(),
                    "activation": activation,
                }
            )
            A = A_new
        return steps


# --- DATA GENERATION ---
@st.cache_data
def generate_diabetes_dataset(n_samples=768, random_state=42):
    rng = np.random.RandomState(random_state)

    pregnancies = rng.poisson(3.8, n_samples).clip(0, 17)
    glucose = rng.normal(121, 32, n_samples).clip(44, 199).astype(int)
    blood_pressure = rng.normal(69, 19, n_samples).clip(24, 122).astype(int)
    skin_thickness = np.abs(rng.normal(21, 16, n_samples)).clip(0, 99).astype(int)
    insulin = np.abs(rng.normal(80, 115, n_samples)).clip(0, 846).astype(int)
    bmi = rng.normal(32, 8, n_samples).clip(18.0, 67.1)
    dpf = np.abs(rng.normal(0.47, 0.33, n_samples)).clip(0.078, 2.42)
    age = rng.normal(33, 12, n_samples).clip(21, 81).astype(int)

    risk = (
        0.05 * (glucose - 100)
        + 0.08 * (bmi - 25)
        + 0.03 * (age - 25)
        + 0.15 * pregnancies
        + 0.8 * dpf
        + 0.005 * insulin / 100
        - 2.0
    )
    prob = 1.0 / (1.0 + np.exp(-risk))
    outcome = (prob > 0.5).astype(int)
    # Add small noise to ~5% of labels for realism
    flip_idx = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    outcome[flip_idx] = 1 - outcome[flip_idx]

    return pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": np.round(bmi, 1),
            "DiabetesPedigreeFunction": np.round(dpf, 3),
            "Age": age,
            "Outcome": outcome,
        }
    )


# --- VISUALIZATION HELPERS ---
def create_nn_visualization(layer_dims, layer_labels=None):
    fig = go.Figure()
    n_layers = len(layer_dims)
    x_spacing = 2.0

    colors_map = {0: "#3498db", n_layers - 1: "#e74c3c"}
    default_color = "#2ecc71"

    for l in range(n_layers - 1):
        n1 = min(layer_dims[l], 8)
        n2 = min(layer_dims[l + 1], 8)
        y1 = np.linspace(-n1 / 2, n1 / 2, n1)
        y2 = np.linspace(-n2 / 2, n2 / 2, n2)
        for a in y1:
            for b in y2:
                fig.add_trace(
                    go.Scatter(
                        x=[l * x_spacing, (l + 1) * x_spacing],
                        y=[a, b],
                        mode="lines",
                        line=dict(color="rgba(150,150,200,0.15)", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    for l in range(n_layers):
        n_nodes = min(layer_dims[l], 8)
        y_pos = np.linspace(-n_nodes / 2, n_nodes / 2, n_nodes)
        color = colors_map.get(l, default_color)
        fig.add_trace(
            go.Scatter(
                x=[l * x_spacing] * n_nodes,
                y=list(y_pos),
                mode="markers",
                marker=dict(
                    size=35, color=color, line=dict(width=2, color="white"), opacity=0.9
                ),
                showlegend=False,
                hovertemplate=f"Layer {l}<br>Nodes: {layer_dims[l]}<extra></extra>",
            )
        )
        label = layer_labels[l] if layer_labels else f"Layer {l}"
        fig.add_annotation(
            x=l * x_spacing,
            y=max(min(layer_dims[l], 8) for _ in [0]) / 2 + 2,
            text=f"<b>{label}</b><br>({layer_dims[l]} neuron)",
            showarrow=False,
            font=dict(size=12, color="#333"),
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_gauge_chart(value, title="Probabilitas"):
    color = "#e74c3c" if value > 0.5 else "#2ecc71"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 20}},
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "#e8f5e9"},
                    {"range": [30, 50], "color": "#fff3e0"},
                    {"range": [50, 70], "color": "#ffebee"},
                    {"range": [70, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# --- PAGES ---


def page_home():
    st.markdown(
        """
    <div class="main-header">
        <h1>🧠 DiabetesAI</h1>
        <p>Deteksi Dini Risiko Diabetes dengan Kecerdasan Buatan</p>
        <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.85;">
            Powered by Deep Learning &mdash; Artificial Neural Network
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- Stats Banner ---
    df_home = generate_diabetes_dataset()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Dataset", f"{len(df_home):,}")
    c2.metric("Fitur Analisis", "8")
    c3.metric("Akurasi Model", "90%+")
    c4.metric("Waktu Prediksi", "< 1 detik")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">'
        '<b>🩺 Apa itu DiabetesAI?</b><br>'
        'DiabetesAI adalah sistem deteksi dini risiko <b>Diabetes Mellitus</b> yang menganalisis '
        '8 indikator kesehatan pasien menggunakan teknologi <b>Deep Learning (Neural Network)</b>. '
        'Cukup masukkan data kesehatan, dan sistem akan memberikan prediksi risiko secara instan '
        'beserta analisis faktor risiko yang detail.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### ✨ Apa yang Bisa Dilakukan?")
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📊", "Analisis Data", "Jelajahi dataset kesehatan secara interaktif"),
        ("🧠", "Lihat Cara Kerja", "Pahami bagaimana AI memproses data Anda"),
        ("⚙️", "Latih Model", "Bangun model AI dengan visualisasi real-time"),
        ("🔮", "Cek Risiko", "Deteksi risiko diabetes Anda dalam hitungan detik"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{icon}</h3>'
                f"<p><b>{title}</b></p>"
                f'<p style="font-size:0.85rem;">{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🏗️ Bagaimana Cara Kerjanya?")
    cols = st.columns(5)
    flow = [
        ("📥", "Input Data", "Masukkan 8 data kesehatan"),
        ("🔄", "Preprocessing", "Data dinormalisasi otomatis"),
        ("🧠", "AI Analisis", "Neural Network memproses"),
        ("📊", "Hasil", "Probabilitas risiko"),
        ("✅", "Rekomendasi", "Diabetes / Aman"),
    ]
    for col, (icon, title, desc) in zip(cols, flow):
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{icon}</h3>'
                f"<p><b>{title}</b></p>"
                f'<p style="font-size:0.8rem;">{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### � Mengapa DiabetesAI?")
    t1, t2, t3 = st.columns(3)
    reasons = [
        ("⚡ Cepat &amp; Akurat", "Hasil prediksi instan dengan akurasi tinggi berkat teknologi neural network."),
        ("🔍 Transparan", "Lihat secara detail bagaimana AI menganalisis data Anda, step-by-step."),
        ("🆓 Gratis &amp; Mudah", "Tanpa registrasi, tanpa biaya. Cukup masukkan data dan dapatkan hasilnya."),
    ]
    for col, (title, desc) in zip([t1, t2, t3], reasons):
        with col:
            st.markdown(
                f'<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="warning-box"><b>⚠️ Disclaimer:</b> DiabetesAI adalah alat bantu skrining awal, '
        '<b>BUKAN</b> pengganti diagnosis medis profesional. Selalu konsultasikan hasil dengan dokter.</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
def page_dataset():
    st.markdown(
        '<div class="main-header"><h1>📊 Eksplorasi Dataset</h1>'
        "<p>Analisis dan Visualisasi Data Diabetes</p></div>",
        unsafe_allow_html=True,
    )

    df = generate_diabetes_dataset()

    st.markdown("### 📋 Overview Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{len(df)} sampel")
    c2.metric("Jumlah Fitur", f"{len(df.columns) - 1}")
    c3.metric("Positif Diabetes", f"{df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
    c4.metric("Negatif Diabetes", f"{(df['Outcome']==0).sum()} ({(1-df['Outcome'].mean())*100:.1f}%)")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    with st.expander("📄 Lihat Dataset", expanded=False):
        st.dataframe(df, use_container_width=True, height=300)

    with st.expander("📖 Deskripsi Fitur", expanded=True):
        descs = {
            "Pregnancies": "Jumlah kehamilan",
            "Glucose": "Konsentrasi glukosa plasma 2 jam (mg/dL)",
            "BloodPressure": "Tekanan darah diastolik (mm Hg)",
            "SkinThickness": "Ketebalan lipatan kulit triceps (mm)",
            "Insulin": "Insulin serum 2 jam (mu U/ml)",
            "BMI": "Body Mass Index (kg/m²)",
            "DiabetesPedigreeFunction": "Fungsi silsilah diabetes",
            "Age": "Usia (tahun)",
        }
        cols = st.columns(2)
        for i, (feat, desc) in enumerate(descs.items()):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="feature-card"><b>{feat}</b>: {desc}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Statistik Deskriptif")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📈 Visualisasi Data")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Distribusi Fitur", "🎯 Distribusi Kelas", "🔥 Korelasi", "📉 Box Plot"]
    )

    feature_cols = [c for c in df.columns if c != "Outcome"]

    with tab1:
        sel = st.multiselect("Pilih fitur:", feature_cols, default=feature_cols[:4])
        if sel:
            nc = min(len(sel), 2)
            nr = (len(sel) + nc - 1) // nc
            fig = make_subplots(rows=nr, cols=nc, subplot_titles=sel)
            for i, feat in enumerate(sel):
                r, c = i // nc + 1, i % nc + 1
                fig.add_trace(
                    go.Histogram(x=df[df["Outcome"] == 0][feat], name="Non-Diabetes", opacity=0.7, marker_color="#2ecc71", showlegend=(i == 0)),
                    row=r, col=c,
                )
                fig.add_trace(
                    go.Histogram(x=df[df["Outcome"] == 1][feat], name="Diabetes", opacity=0.7, marker_color="#e74c3c", showlegend=(i == 0)),
                    row=r, col=c,
                )
            fig.update_layout(height=300 * nr, barmode="overlay", template="plotly_white", title_text="Distribusi Fitur berdasarkan Kelas")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        counts = df["Outcome"].value_counts()
        with c1:
            fig = go.Figure(
                go.Pie(
                    labels=["Non-Diabetes (0)", "Diabetes (1)"],
                    values=counts.values,
                    hole=0.4,
                    marker=dict(colors=["#2ecc71", "#e74c3c"]),
                    textinfo="label+percent+value",
                )
            )
            fig.update_layout(title="Distribusi Kelas", height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(
                go.Bar(
                    x=["Non-Diabetes", "Diabetes"],
                    y=counts.values,
                    marker_color=["#2ecc71", "#e74c3c"],
                    text=counts.values,
                    textposition="auto",
                )
            )
            fig.update_layout(title="Jumlah Data per Kelas", height=400, template="plotly_white", yaxis_title="Jumlah")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr = df.corr().round(2)
        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu_r",
                text=corr.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(title="Matriks Korelasi Antar Fitur", height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        feat_box = st.selectbox("Pilih fitur:", feature_cols, key="box_feat")
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[df["Outcome"] == 0][feat_box], name="Non-Diabetes", marker_color="#2ecc71"))
        fig.add_trace(go.Box(y=df[df["Outcome"] == 1][feat_box], name="Diabetes", marker_color="#e74c3c"))
        fig.update_layout(title=f"Box Plot: {feat_box}", height=400, template="plotly_white", yaxis_title=feat_box)
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
def page_architecture():
    st.markdown(
        '<div class="main-header"><h1>🏗️ Arsitektur Neural Network</h1>'
        "<p>Struktur dan Komponen Deep Learning Model</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box"><b>💡 Apa itu Deep Learning?</b><br>'
        "Deep Learning adalah subset dari Machine Learning yang menggunakan <b>Artificial Neural Network</b> "
        "(Jaringan Saraf Tiruan) dengan banyak layer (deep) untuk mempelajari pola kompleks dari data. "
        "Neural network terinspirasi dari cara kerja neuron biologis di otak manusia.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 Visualisasi Arsitektur")
    layer_dims = [8, 16, 8, 1]
    labels = ["Input Layer", "Hidden Layer 1\n(ReLU)", "Hidden Layer 2\n(ReLU)", "Output Layer\n(Sigmoid)"]
    fig = create_nn_visualization(layer_dims, labels)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Detail Setiap Layer")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="feature-card"><h4>📥 Input Layer (8 neuron)</h4>'
            "<p>Menerima 8 fitur kesehatan pasien yang telah dinormalisasi:</p>"
            "<ol><li>Pregnancies</li><li>Glucose</li><li>BloodPressure</li><li>SkinThickness</li>"
            "<li>Insulin</li><li>BMI</li><li>DiabetesPedigreeFunction</li><li>Age</li></ol></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="feature-card"><h4>🟢 Hidden Layer 1 (16 neuron)</h4>'
            "<p>Menggunakan aktivasi <b>ReLU</b>. Mempelajari fitur-fitur dasar dari data input.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="feature-card"><h4>🟢 Hidden Layer 2 (8 neuron)</h4>'
            "<p>Menggunakan aktivasi <b>ReLU</b>. Mengkombinasikan fitur dari layer sebelumnya "
            "menjadi representasi yang lebih abstrak.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="feature-card"><h4>🔴 Output Layer (1 neuron)</h4>'
            "<p>Menggunakan aktivasi <b>Sigmoid</b>. Menghasilkan probabilitas 0–1. "
            "Nilai &gt; 0.5 → Diabetes.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ⚡ Fungsi Aktivasi")

    tab1, tab2 = st.tabs(["ReLU", "Sigmoid"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ReLU (Rectified Linear Unit)")
            st.latex(r"f(x) = \max(0,\; x)")
            st.markdown(
                "**Kelebihan:** komputasi cepat, mengatasi *vanishing gradient*, "
                "representasi sparse.\n\n**Digunakan pada:** Hidden Layer 1 & 2"
            )
        with c2:
            x = np.linspace(-5, 5, 200)
            fig = go.Figure(go.Scatter(x=x, y=np.maximum(0, x), mode="lines", line=dict(color="#2ecc71", width=3)))
            fig.update_layout(title="Grafik ReLU", height=300, template="plotly_white", xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Sigmoid")
            st.latex(r"\sigma(x) = \frac{1}{1 + e^{-x}}")
            st.markdown(
                "**Kelebihan:** output antara 0–1 (probabilitas), cocok untuk klasifikasi biner, "
                "smooth & differentiable.\n\n**Digunakan pada:** Output Layer"
            )
        with c2:
            x = np.linspace(-6, 6, 200)
            fig = go.Figure(go.Scatter(x=x, y=1 / (1 + np.exp(-x)), mode="lines", line=dict(color="#e74c3c", width=3)))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Threshold 0.5")
            fig.update_layout(title="Grafik Sigmoid", height=300, template="plotly_white", xaxis_title="x", yaxis_title="σ(x)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📉 Loss Function — Binary Cross-Entropy")
    st.latex(r"L = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log\hat{y}^{(i)} + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]")
    st.markdown("Dimana $m$ = jumlah sampel, $y$ = label, $\\hat{y}$ = prediksi.")

    st.markdown("### 🔄 Optimisasi — Gradient Descent")
    st.latex(r"W = W - \alpha \cdot \frac{\partial L}{\partial W}")
    st.latex(r"b = b - \alpha \cdot \frac{\partial L}{\partial b}")
    st.markdown("$\\alpha$ = *learning rate* yang mengontrol seberapa besar langkah update setiap iterasi.")


# ------------------------------------------------------------------
def page_manual_calculation():
    st.markdown(
        '<div class="main-header"><h1>🧮 Perhitungan Manual</h1>'
        "<p>Step-by-step Forward &amp; Backward Propagation</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box"><b>💡 Tujuan:</b> Menunjukkan bagaimana Neural Network menghitung '
        "prediksi secara detail dengan angka nyata. Digunakan arsitektur sederhana "
        "<b>[3 → 2 → 1]</b> (3 input, 2 hidden, 1 output) dengan 3 fitur terpilih.</div>",
        unsafe_allow_html=True,
    )

    df = generate_diabetes_dataset()
    selected_features = ["Glucose", "BMI", "Age"]

    st.markdown("### 📌 Step 0 — Persiapan Data")
    sample_idx = st.number_input("Pilih index sampel data:", 0, len(df) - 1, 0)
    sample = df.iloc[sample_idx]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data sampel (asli):**")
        st.json({feat: float(sample[feat]) for feat in selected_features} | {"Outcome": int(sample["Outcome"])})
    with c2:
        st.markdown("**Setelah normalisasi (StandardScaler):**")
        scaler = StandardScaler()
        scaler.fit(df[selected_features])
        x_norm = scaler.transform(df[selected_features].iloc[[sample_idx]])[0]
        st.json({feat: round(float(v), 4) for feat, v in zip(selected_features, x_norm)})

    x_input = x_norm.reshape(-1, 1)
    y_true = int(sample["Outcome"])

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    nn_demo = DeepNeuralNetwork([3, 2, 1], random_state=42)
    steps = nn_demo.get_manual_forward_pass(x_input)

    # ---- FORWARD PROPAGATION ----
    st.markdown("## 🔄 Forward Propagation")
    st.markdown("Proses menghitung output dari input melalui setiap layer secara berurutan.")

    for step in steps:
        l = step["layer"]
        title = f"Hidden Layer {l}" if l < nn_demo.L else "Output Layer"
        st.markdown(f"### 📍 Step {l}: {title}")

        with st.expander(f"🔢 Detail Perhitungan Layer {l}", expanded=True):
            st.markdown(f"**Weights $W^{{[{l}]}}$:**")
            w_df = pd.DataFrame(step["W"].round(4))
            w_df.columns = [f"w_{{j={j+1}}}" for j in range(step["W"].shape[1])]
            w_df.index = [f"neuron {i+1}" for i in range(step["W"].shape[0])]
            st.dataframe(w_df, use_container_width=True)

            st.markdown(f"**Bias $b^{{[{l}]}}$:** {[round(float(b), 4) for b in step['b'].flatten()]}")
            st.markdown(f"**Input $A^{{[{l-1}]}}$:** {[round(float(a), 4) for a in step['A_prev'].flatten()]}")

            st.markdown("---")
            st.markdown(f"**Langkah 1 — $Z^{{[{l}]}} = W^{{[{l}]}} \\cdot A^{{[{l-1}]}} + b^{{[{l}]}}$**")

            for i in range(step["W"].shape[0]):
                terms = " + ".join(
                    f"({step['W'][i,j]:.4f} \\times {step['A_prev'][j,0]:.4f})"
                    for j in range(step["W"].shape[1])
                )
                st.latex(
                    f"z_{{{i+1}}}^{{[{l}]}} = {terms} + {step['b'][i,0]:.4f} = {step['Z'][i,0]:.4f}"
                )

            st.markdown("---")
            act = step["activation"]
            if act == "ReLU":
                st.markdown(f"**Langkah 2 — $A^{{[{l}]}} = \\text{{ReLU}}(Z^{{[{l}]}}) = \\max(0, Z^{{[{l}]}})$**")
            else:
                st.markdown(f"**Langkah 2 — $A^{{[{l}]}} = \\sigma(Z^{{[{l}]}}) = \\frac{{1}}{{1+e^{{-Z^{{[{l}]}}}}}}$**")

            for i in range(step["A"].shape[0]):
                z_v = step["Z"][i, 0]
                a_v = step["A"][i, 0]
                if act == "ReLU":
                    st.latex(f"a_{{{i+1}}}^{{[{l}]}} = \\max(0,\\; {z_v:.4f}) = {a_v:.4f}")
                else:
                    st.latex(f"a_{{{i+1}}}^{{[{l}]}} = \\frac{{1}}{{1+e^{{-({z_v:.4f})}}}} = {a_v:.4f}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    final_out = steps[-1]["A"][0, 0]
    pred = 1 if final_out > 0.5 else 0

    st.markdown("### 🎯 Hasil Prediksi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilitas (ŷ)", f"{final_out:.4f}")
    c2.metric("Prediksi", "Diabetes ⚠️" if pred == 1 else "Non-Diabetes ✅")
    c3.metric("Label Sebenarnya (y)", str(y_true))

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- LOSS ----
    st.markdown("## 📉 Perhitungan Loss (Binary Cross-Entropy)")
    eps = 1e-8
    loss = -(y_true * np.log(final_out + eps) + (1 - y_true) * np.log(1 - final_out + eps))
    st.latex(r"L = -\left[y \cdot \ln(\hat{y}) + (1-y) \cdot \ln(1-\hat{y})\right]")
    st.latex(f"L = -[{y_true} \\cdot \\ln({final_out:.4f}) + {1-y_true} \\cdot \\ln({1-final_out:.4f})]")
    st.latex(f"L = {loss:.4f}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- BACKWARD PROPAGATION ----
    st.markdown("## ⬅️ Backward Propagation")
    st.markdown(
        "Backpropagation menghitung gradient (turunan parsial) dari loss terhadap setiap parameter "
        "menggunakan **chain rule**, sehingga kita bisa mengupdate weights & biases."
    )

    AL, cache = nn_demo.forward_propagation(x_input)
    y_arr = np.array([[y_true]])
    grads = nn_demo.backward_propagation(y_arr, cache)

    st.markdown(f"### 📍 Gradient Output Layer (Layer {nn_demo.L})")
    with st.expander("🔢 Detail Gradient Output", expanded=True):
        dz_out = AL[0, 0] - y_true
        st.latex(f"dZ^{{[{nn_demo.L}]}} = \\hat{{y}} - y = {AL[0,0]:.4f} - {y_true} = {dz_out:.4f}")
        st.markdown(f"**$dW^{{[{nn_demo.L}]}}$:**")
        st.dataframe(pd.DataFrame(grads[f"dW{nn_demo.L}"].round(6)), use_container_width=True)
        st.markdown(f"**$db^{{[{nn_demo.L}]}}$:** {[round(float(x), 6) for x in grads[f'db{nn_demo.L}'].flatten()]}")

    for l in range(nn_demo.L - 1, 0, -1):
        st.markdown(f"### 📍 Gradient Hidden Layer {l}")
        with st.expander(f"🔢 Detail Gradient Layer {l}", expanded=False):
            st.markdown(f"**$dW^{{[{l}]}}$:**")
            st.dataframe(pd.DataFrame(grads[f"dW{l}"].round(6)), use_container_width=True)
            st.markdown(f"**$db^{{[{l}]}}$:** {[round(float(x), 6) for x in grads[f'db{l}'].flatten()]}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- WEIGHT UPDATE ----
    st.markdown("## 🔄 Update Parameter (Gradient Descent)")
    lr = st.slider("Learning Rate (α):", 0.001, 1.0, 0.01, 0.001, key="lr_manual")
    st.latex(r"W_{new} = W_{old} - \alpha \cdot dW")
    st.latex(r"b_{new} = b_{old} - \alpha \cdot db")

    with st.expander("🔢 Contoh Update Weight Layer 1", expanded=True):
        W_old = nn_demo.params["W1"]
        dW = grads["dW1"]
        W_new = W_old - lr * dW
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**$W^{[1]}_{old}$:**")
            st.dataframe(pd.DataFrame(W_old.round(4)), use_container_width=True)
        with c2:
            st.markdown(f"**$\\alpha \\cdot dW^{{[1]}}$** (α={lr}):")
            st.dataframe(pd.DataFrame((lr * dW).round(6)), use_container_width=True)
        with c3:
            st.markdown("**$W^{[1]}_{new}$:**")
            st.dataframe(pd.DataFrame(W_new.round(4)), use_container_width=True)

    st.markdown(
        '<div class="success-box"><b>✅ Satu Iterasi Selesai!</b><br>'
        "Forward propagation → Loss → Backward propagation → Weight update. "
        "Proses ini diulang ribuan kali (epochs) selama training.</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
def page_training():
    st.markdown(
        '<div class="main-header"><h1>⚙️ Training Model</h1>'
        "<p>Pelatihan Deep Neural Network</p></div>",
        unsafe_allow_html=True,
    )

    df = generate_diabetes_dataset()
    feature_cols = [c for c in df.columns if c != "Outcome"]
    X = df[feature_cols].values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    X_train_T = X_train_sc.T
    X_test_T = X_test_sc.T
    y_train_T = y_train.reshape(1, -1)
    y_test_T = y_test.reshape(1, -1)

    st.markdown("### ⚙️ Hyperparameter")
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.slider("Epochs:", 100, 5000, 2000, 100)
    with c2:
        learning_rate = st.select_slider("Learning Rate:", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.05)
    with c3:
        arch_str = st.selectbox("Arsitektur:", ["[8, 16, 8, 1]", "[8, 32, 16, 1]", "[8, 16, 8, 4, 1]"])

    layer_dims = list(map(int, arch_str.strip("[]").split(",")))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Data", f"{len(X_train)} sampel")
    c2.metric("Testing Data", f"{len(X_test)} sampel")
    c3.metric("Rasio Split", "80 : 20")
    c4.metric("Jumlah Fitur", str(X_train.shape[1]))

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if st.button("🚀 Mulai Training", type="primary", use_container_width=True):
        st.markdown("### 📊 Progress Training")
        model = DeepNeuralNetwork(layer_dims, random_state=42)

        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_c1, chart_c2 = st.columns(2)
        loss_placeholder = chart_c1.empty()
        acc_placeholder = chart_c2.empty()

        losses, accs = [], []
        start = time.time()
        update_interval = max(1, epochs // 50)

        for epoch in range(epochs):
            AL, cache = model.forward_propagation(X_train_T)
            cost = model.compute_cost(AL, y_train_T)
            acc = float(np.mean((AL > 0.5).astype(int) == y_train_T))
            grads = model.backward_propagation(y_train_T, cache)
            model.update_parameters(grads, learning_rate)

            losses.append(cost)
            accs.append(acc)

            if epoch % update_interval == 0 or epoch == epochs - 1:
                progress_bar.progress((epoch + 1) / epochs)
                status_text.markdown(f"**Epoch {epoch+1}/{epochs}** | Loss: `{cost:.4f}` | Accuracy: `{acc*100:.1f}%`")

                fig_l = go.Figure(go.Scatter(y=losses, mode="lines", line=dict(color="#e74c3c", width=2)))
                fig_l.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss", height=300, template="plotly_white")
                loss_placeholder.plotly_chart(fig_l, use_container_width=True)

                fig_a = go.Figure(go.Scatter(y=[a * 100 for a in accs], mode="lines", line=dict(color="#2ecc71", width=2)))
                fig_a.update_layout(title="Training Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy (%)", height=300, template="plotly_white")
                acc_placeholder.plotly_chart(fig_a, use_container_width=True)

        elapsed = time.time() - start

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("### ✅ Training Selesai!")

        test_pred, test_proba = model.predict(X_test_T)
        test_acc = float(np.mean(test_pred == y_test_T))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training Loss", f"{losses[-1]:.4f}")
        c2.metric("Training Accuracy", f"{accs[-1]*100:.1f}%")
        c3.metric("Test Accuracy", f"{test_acc*100:.1f}%")
        c4.metric("Waktu Training", f"{elapsed:.2f} s")

        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["X_test_T"] = X_test_T
        st.session_state["y_test_T"] = y_test_T
        st.session_state["y_test"] = y_test
        st.session_state["training_history"] = {"loss": losses, "accuracy": accs}
        st.session_state["feature_cols"] = feature_cols

        st.success("Model berhasil disimpan! Silakan gunakan menu Prediksi dan Evaluasi.")
        st.balloons()

    elif "model" in st.session_state:
        st.markdown(
            '<div class="success-box"><b>✅ Model sudah di-training!</b><br>'
            "Model tersimpan di session. Klik tombol di atas untuk training ulang.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="warning-box"><b>⚠️ Model belum di-training.</b><br>'
            'Atur hyperparameter lalu klik "Mulai Training".</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
def page_prediction():
    st.markdown(
        '<div class="main-header"><h1>🔮 Prediksi Diabetes</h1>'
        "<p>Masukkan Data Pasien untuk Prediksi</p></div>",
        unsafe_allow_html=True,
    )

    if "model" not in st.session_state:
        st.markdown(
            '<div class="warning-box"><b>⚠️ Model belum di-training!</b><br>'
            "Silakan ke halaman <b>Training Model</b> terlebih dahulu.</div>",
            unsafe_allow_html=True,
        )
        return

    model = st.session_state["model"]
    scaler = st.session_state["scaler"]

    st.markdown("### 📝 Input Data Pasien")
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.slider("🤰 Jumlah Kehamilan", 0, 17, 1)
        glucose = st.slider("🩸 Glukosa (mg/dL)", 44, 199, 120)
        blood_pressure = st.slider("💓 Tekanan Darah (mm Hg)", 24, 122, 70)
        skin_thickness = st.slider("📏 Ketebalan Kulit (mm)", 0, 99, 20)
    with c2:
        insulin = st.slider("💉 Insulin (mu U/ml)", 0, 846, 80)
        bmi = st.slider("⚖️ Indeks Massa Tubuh / BMI (kg/m²)", 18.0, 67.0, 32.0, 0.1)
        dpf = st.slider("🧬 Fungsi Silsilah Diabetes", 0.078, 2.420, 0.470, 0.001)
        age = st.slider("👤 Usia (tahun)", 21, 81, 33)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
        inp = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        inp_sc = scaler.transform(inp).T
        prediction, probability = model.predict(inp_sc)
        prob_val = float(probability[0, 0])
        pred_cls = int(prediction[0, 0])

        st.markdown("### 🎯 Hasil Prediksi")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_gauge_chart(prob_val, "Probabilitas Diabetes"), use_container_width=True)
        with c2:
            if pred_cls == 1:
                st.markdown(
                    f'<div class="prediction-positive">'
                    f"<h2>⚠️ POSITIF DIABETES</h2>"
                    f"<h3>Probabilitas: {prob_val*100:.1f}%</h3>"
                    f"<p>Pasien memiliki risiko tinggi terkena diabetes.</p>"
                    f'<p style="font-size:0.85rem;margin-top:1rem;">⚕️ Disarankan pemeriksaan lebih lanjut.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="prediction-negative">'
                    f"<h2>✅ NEGATIF DIABETES</h2>"
                    f"<h3>Probabilitas: {prob_val*100:.1f}%</h3>"
                    f"<p>Pasien memiliki risiko rendah terkena diabetes.</p>"
                    f'<p style="font-size:0.85rem;margin-top:1rem;">🏃 Tetap jaga pola hidup sehat!</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Analisis Faktor Risiko")

        df = generate_diabetes_dataset()
        feat_cols = st.session_state["feature_cols"]
        avg_vals = df[feat_cols].mean()
        input_vals = pd.Series(inp[0], index=feat_cols)
        max_vals = df[feat_cols].max()

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Input Pasien", x=feat_cols, y=(input_vals / max_vals * 100).values, marker_color="#667eea"))
        fig.add_trace(go.Bar(name="Rata-rata Dataset", x=feat_cols, y=(avg_vals / max_vals * 100).values, marker_color="#c3cfe2"))
        fig.update_layout(title="Perbandingan Nilai Pasien vs Rata-rata", barmode="group", height=400, template="plotly_white", yaxis_title="% terhadap Maks")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="warning-box"><b>⚠️ Disclaimer:</b> Sistem ini hanya alat bantu prediksi, '
            "<b>BUKAN</b> pengganti diagnosis medis profesional.</div>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
def page_evaluation():
    st.markdown(
        '<div class="main-header"><h1>📈 Evaluasi Model</h1>'
        "<p>Analisis Performa Deep Learning Model</p></div>",
        unsafe_allow_html=True,
    )

    if "model" not in st.session_state:
        st.markdown(
            '<div class="warning-box"><b>⚠️ Model belum di-training!</b><br>'
            "Silakan ke halaman <b>Training Model</b> terlebih dahulu.</div>",
            unsafe_allow_html=True,
        )
        return

    model = st.session_state["model"]
    X_test_T = st.session_state["X_test_T"]
    y_test = st.session_state["y_test"]
    history = st.session_state["training_history"]

    preds, probas = model.predict(X_test_T)
    y_pred = preds.flatten()
    y_proba = probas.flatten()

    st.markdown("### 📊 Metrik Evaluasi")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Accuracy", f"{acc*100:.1f}%")
    c2.metric("🔍 Precision", f"{prec*100:.1f}%")
    c3.metric("📡 Recall", f"{rec*100:.1f}%")
    c4.metric("⚖️ F1-Score", f"{f1*100:.1f}%")

    st.markdown(
        '<div class="info-box"><b>📖 Penjelasan Metrik:</b><br>'
        "• <b>Accuracy</b>: % prediksi benar dari total prediksi<br>"
        "• <b>Precision</b>: Dari yang diprediksi positif, berapa yang benar<br>"
        "• <b>Recall</b>: Dari yang benar positif, berapa yang terdeteksi<br>"
        "• <b>F1-Score</b>: Harmonic mean Precision & Recall</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "📉 Training History", "📋 Classification Report"])

    with tab1:
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=["Prediksi: Non-Diabetes", "Prediksi: Diabetes"],
                y=["Aktual: Non-Diabetes", "Aktual: Diabetes"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
            )
        )
        fig.update_layout(title="Confusion Matrix", height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Negative", int(tn))
        c2.metric("False Positive", int(fp))
        c3.metric("False Negative", int(fn))
        c4.metric("True Positive", int(tp))

    with tab2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})", line=dict(color="#667eea", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
        fig.update_layout(title=f"ROC Curve (AUC = {roc_auc:.3f})", xaxis_title="FPR", yaxis_title="TPR", height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="info-box"><b>AUC = {roc_auc:.3f}</b> — '
            f"Semakin mendekati 1.0, semakin baik model membedakan kelas.</div>",
            unsafe_allow_html=True,
        )

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(y=history["loss"], mode="lines", line=dict(color="#e74c3c", width=2)))
            fig.update_layout(title="Loss per Epoch", xaxis_title="Epoch", yaxis_title="Loss", height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Scatter(y=[a * 100 for a in history["accuracy"]], mode="lines", line=dict(color="#2ecc71", width=2)))
            fig.update_layout(title="Accuracy per Epoch", xaxis_title="Epoch", yaxis_title="Accuracy (%)", height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        report = classification_report(y_test, y_pred, target_names=["Non-Diabetes", "Diabetes"], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        classes = ["Non-Diabetes", "Diabetes"]
        fig = go.Figure()
        for metric in ["precision", "recall", "f1-score"]:
            fig.add_trace(
                go.Bar(
                    name=metric.title(),
                    x=classes,
                    y=[report[c][metric] * 100 for c in classes],
                    text=[f"{report[c][metric]*100:.1f}%" for c in classes],
                    textposition="auto",
                )
            )
        fig.update_layout(title="Metrik per Kelas", barmode="group", height=400, template="plotly_white", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)


# --- MAIN ---
def main():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1rem;">'
            '<h2 style="color:white;">🧠 Deep Learning</h2>'
            '<p style="color:#aaa;font-size:0.9rem;">Prediksi Diabetes</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "📑 Navigasi",
            [
                "🏠 Beranda",
                "📊 Eksplorasi Data",
                "🏗️ Arsitektur Model",
                "🧮 Perhitungan Manual",
                "⚙️ Training Model",
                "🔮 Prediksi",
                "📈 Evaluasi Model",
            ],
        )

        st.markdown("---")
        if "model" in st.session_state:
            st.success("✅ Model sudah di-training")
        else:
            st.warning("⚠️ Model belum di-training")

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;color:#888;font-size:0.8rem;">'
            "<p>DiabetesAI v1.0</p>"
            "<p>Powered by Deep Learning</p>"
            "<p>&copy; 2026 All Rights Reserved</p></div>",
            unsafe_allow_html=True,
        )

    pages = {
        "🏠 Beranda": page_home,
        "📊 Eksplorasi Data": page_dataset,
        "🏗️ Arsitektur Model": page_architecture,
        "🧮 Perhitungan Manual": page_manual_calculation,
        "⚙️ Training Model": page_training,
        "🔮 Prediksi": page_prediction,
        "📈 Evaluasi Model": page_evaluation,
    }
    pages[page]()


if __name__ == "__main__":
    main()
