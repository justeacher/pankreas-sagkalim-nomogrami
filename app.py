import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Sayfa ayarları - Sol panel varsayılan olarak açık başlar
st.set_page_config(
    page_title="Pancreatic Cancer CDSS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TAM BEYAZ TEMA ZORLAYICI CSS (TÜM BİLEŞENLER İÇİN) ---
st.markdown("""
    <style>
        /* Tarayıcı ve kök bileşenleri açık temaya zorla */
        :root {
            color-scheme: light !important;
        }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        /* Sidebar arka planı */
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #CBD5E1 !important;
        }
        /* Üst boşluk ayarı (başlıkların kesilmemesi için) */
        .block-container {
            padding-top: 2.2rem !important;
            padding-bottom: 2rem !important;
        }
        /* Tüm metinlerin koyu renkte net görünmesi */
        h1, h2, h3, h4, p, span, label, div[data-testid="stMarkdownContainer"] {
            color: #0F172A !important;
        }
        /* Selectbox (Açılır kutular) ve içindeki metinler */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #94A3B8 !important;
            min-height: 36px !important;
        }
        div[data-baseweb="select"] span, div[data-baseweb="select"] svg {
            color: #0F172A !important;
            fill: #0F172A !important;
        }
        ul[role="listbox"], ul[role="listbox"] li {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        /* Expander (Katlanabilir kutular - Hem Sidebar hem Alt Tablo) */
        div[data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stExpander"] details summary {
            background-color: #F1F5F9 !important;
            color: #0F172A !important;
            border-radius: 8px !important;
        }
        div[data-testid="stExpander"] details summary span,
        div[data-testid="stExpander"] details summary svg {
            color: #0F172A !important;
            fill: #0F172A !important;
        }
        /* Metrik kartları */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
        }
        /* Uyarı (st.warning / st.info) kutularını açık sarı/mavi yapma */
        div[data-testid="stAlert"] {
            background-color: #FEF9C3 !important;
            color: #0F172A !important;
            border: 1px solid #EAB308 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stAlert"] p, div[data-testid="stAlert"] strong {
            color: #0F172A !important;
        }
        /* Buton stili */
        div.stButton > button {
            background-color: #0F172A !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: none !important;
            width: 100%;
            font-weight: 600;
        }
        div.stButton > button p {
            color: #FFFFFF !important;
        }
        /* Dataframe / Tablo arka planı */
        [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
        }
        /* Grafik kapsayıcısı */
        div[data-testid="stVegaLiteChart"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SÖZLÜK ---
D = {
    "Türkçe": {
        "title": "🩺 Pankreas Kanseri Sağkalım Klinik Karar Destek Sistemi",
        "subtitle": "Parametrik (Log-logistic AFT), Makine Öğrenmesi (RSF) ve Derin Öğrenme (DeepAFT-LL) Karşılaştırmalı Nomogramı",
        "sidebar_header": "📋 Hasta & Klinik Parametreler",
        "demo_sec": "👤 Demografik Bilgiler",
        "clin_sec": "🔬 Tümör & Tedavi Bilgileri",
        "sex": "Cinsiyet", "race": "Irk", "marital": "Medeni Durum", "age": "Yaş Grubu", "treatment": "Tedavi Protokolü",
        "site": "Tümör Yerleşimi", "grade": "Histolojik Grup", "lnr": "Lenf Nodu Oranı (LNR)", "stage": "Klinik Evre", "t_cat": "Tümör Boyutu (T)",
        "slider_lbl": "Hedef Takip Süresi (Ay):",
        "btn_lbl": "🚀 Analizi Güncelle / Hesapla",
        "aft_lbl": "📊 Log-logistic AFT",
        "rsf_lbl": "🌲 RSF (En İyi AUC)",
        "deep_lbl": "🧠 DeepAFT-LL",
        "med_lbl": "Medyan Sağkalım",
        "chart_title": "📈 Zamana Bağlı Sağkalım Eğrileri (0–120 Ay)",
        "zoom_lbl": "🔍 Eğri Farklarını Yakınlaştır (Otomatik Y-Ekseni)",
        "x_axis": "Zaman (Ay)",
        "y_axis": "Sağkalım Olasılığı (%)",
        "model_lbl": "Model",
        "report_title": "💡 Klinik Bulguların Değerlendirilmesi",
        "perf_title": "📊 Tez Model Performans Özeti (Benchmark Tablosu)",
        "sidebar_info": "💡 Genel ayırt edicilikte (C-İndeksi) **Log-Lojistik AFT (0.812)**, derin öğrenme modelleri arasında **DeepAFT-LL (0.784)** ve uzun vadeli başarıda (AUC) **RSF** öne çıkmaktadır.",
        "risk_lbl": "Hasta Risk Sınıfı",
        "summary_lbl": "Prognostik Özet",
        "high_risk": "🔴 YÜKSEK RİSK GRUBU",
        "mid_risk": "🟡 ORTA RİSK GRUBU",
        "low_risk": "🟢 DÜŞÜK RİSK GRUBU",
        "high_txt": "Hasta kısa dönemli yüksek sağkalım riskine sahiptir. Multidisipliner yaklaşım, agresif tedavi protokolleri ve yakın klinik takip önerilmektedir.",
        "mid_txt": "Standart kombinasyon/adjuvan tedavileriyle hastanın sağkalım eğrisinin stabilize edilmesi hedeflenmelidir.",
        "low_txt": "Genel pankreas kanseri kohortuna kıyasla daha olumlu bir prognostik seyir öngörülmektedir.",
        "outcome_txt": "Çoklu model simülasyonuna göre tahmini medyan sağkalım süresi **{median:.2f} ay** olarak hesaplanmıştır. Seçilen **{month}. ayda** hastanın sağkalım olasılığı modeller arasında tutarlı bir seyir izlemektedir.",
        "disclaimer": "⚠️ **YASAL UYARI:** Bu araç yalnızca akademik araştırma amaçlıdır; doğrudan klinik karar mekanizması olarak kullanılamaz."
    },
    "English": {
        "title": "🩺 Pancreatic Cancer Survival Clinical Decision Support System",
        "subtitle": "Comparative Nomogram of Parametric (Log-logistic AFT), Machine Learning (RSF), and Deep Learning (DeepAFT-LL)",
        "sidebar_header": "📋 Patient & Clinical Parameters",
        "demo_sec": "👤 Demographics",
        "clin_sec": "🔬 Tumor & Treatment",
        "sex": "Sex", "race": "Race", "marital": "Marital Status", "age": "Age Group", "treatment": "Therapeutic Protocol",
        "site": "Tumor Site", "grade": "Histologic Group", "lnr": "Lymph Node Ratio (LNR)", "stage": "Clinical Stage", "t_cat": "Tumor Size (T)",
        "slider_lbl": "Target Follow-up Month (t):",
        "btn_lbl": "🚀 Update / Run Analysis",
        "aft_lbl": "📊 Log-logistic AFT",
        "rsf_lbl": "🌲 RSF (Best AUC)",
        "deep_lbl": "🧠 DeepAFT-LL",
        "med_lbl": "Median Survival",
        "chart_title": "📈 Time-Dependent Survival Curves (0–120 Months)",
        "zoom_lbl": "🔍 Zoom In on Curve Differences (Auto Y-Axis)",
        "x_axis": "Time (Months)",
        "y_axis": "Survival Probability (%)",
        "model_lbl": "Model",
        "report_title": "💡 Clinical Findings Evaluation",
        "perf_title": "📊 Thesis Model Performance Benchmark Table",
        "sidebar_info": "💡 **Log-logistic AFT (0.812)** excels in global discrimination (C-Index), **DeepAFT-LL (0.784)** leads among deep learning models, while **RSF** outperforms in long-term success (AUC).",
        "risk_lbl": "Patient Risk Stratification",
        "summary_lbl": "Prognostic Summary",
        "high_risk": "🔴 HIGH RISK GROUP",
        "mid_risk": "🟡 INTERMEDIATE RISK GROUP",
        "low_risk": "🟢 LOW RISK GROUP",
        "high_txt": "The patient carries a high risk for short-term survival. Multidisciplinary approach, aggressive therapeutic intervention, and close clinical surveillance are recommended.",
        "mid_txt": "Therapeutic stabilization of the patient's survival curve should be targeted with standard combination/adjuvant regimens.",
        "low_txt": "A more favorable prognostic trajectory is projected compared to the overall pancreatic cancer cohort.",
        "outcome_txt": "Based on multi-model simulation, the estimated median survival time is **{median:.2f} months**. At the selected **{month}th month**, the patient's survival probability remains consistent across models.",
        "disclaimer": "⚠️ **LEGAL DISCLAIMER:** This tool is for academic research purposes only and cannot be used as a direct clinical decision mechanism."
    }
}

# --- DİL SEÇİMİ VE YAN MENÜ ---
with st.sidebar:
    lang = st.selectbox("🌐 Language / Dil", ["Türkçe", "English"])
    T = D[lang]
    st.markdown(f"### {T['sidebar_header']}")

    with st.expander(T["demo_sec"], expanded=True):
        sex = st.selectbox(T["sex"], ["Erkek / Male", "Kadın / Female"])
        race = st.selectbox(T["race"], ["Beyaz / White", "Siyah / Black", "Diğer / Other"])
        marital = st.selectbox(T["marital"], ["Evli / Married", "Bekar / Single", "Dul-Boşanmış / Widowed-Divorced"])
        age = st.selectbox(T["age"], ["< 50", "50 - 59", "60 - 69", "70 - 79", "80+"])
        treatment = st.selectbox(T["treatment"], [
            "Cerrahi + Kemoterapi / Surgery + Chemo",
            "Sadece Cerrahi / Surgery Only",
            "Sadece Kemoterapi / Chemo Only",
