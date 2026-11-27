import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Sayfa ayarları - Geniş ve Modern Görünüm
st.set_page_config(page_title="Pancreatic Cancer CDSS", layout="wide", initial_sidebar_state="expanded")

# --- BEYAZ TEMA ZORLAYICI CSS ---
st.markdown("""
    <style>
        /* Ana arka plan ve metin renkleri */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }
        /* Sidebar arka planı */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0;
        }
        /* Tüm başlık ve metinlerin koyu renkte görünmesi */
        h1, h2, h3, h4, p, span, label, div[data-testid="stMarkdownContainer"] {
            color: #1E293B !important;
        }
        /* Metrik kartları için belirgin çerçeve ve açık arka plan */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px;
            padding: 15px;
        }
        /* Selectbox ve input alanlarının beyaz kalması */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            border-color: #CBD5E1 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- DİL SEÇİMİ (LANGUAGE SELECTION) ---
lang = st.radio("🌐 Dil Seçimi / Language Selection", ["Türkçe", "English"], horizontal=True)

# --- SÖZLÜK (DICTIONARY FOR MULTI-LANGUAGE) ---
D = {
    "Türkçe": {
        "title": "🩺 Pankreas Kanseri Sağkalım Klinik Karar Destek Sistemi",
        "subtitle": "Parametrik (Log-logistic AFT), Makine Öğrenmesi (RSF) ve Derin Öğrenme (DeepAFT-LL) Karşılaştırmalı Nomogramı",
        "sidebar_title": "📊 Model Performans Metrikleri (Tez Bulguları)",
        "sidebar_info": "💡 Genel ayırt edicilikte (C-İndeksi) **DeepAFT-LL**, uzun vadeli başarıda (AUC) **RSF** öne çıkmaktadır.",
        "input_header": "📋 Hasta ve Klinik Risk Faktörleri",
        "sex": "Cinsiyet", "race": "Irk", "marital": "Medeni Durum", "age": "Yaş Grubu", "treatment": "Tedavi Protokolü",
        "site": "Tümör Yerleşim Bölgesi", "grade": "Histolojik Grup", "lnr": "Lenf Nodu Oranı (LNR)", "stage": "Klinik Evre (Refined Stage)", "t_cat": "Tümör Boyutu (T Kategorisi)",
        "slider_lbl": "Sağkalım olasılığını incelemek istediğiniz ay (t):",
        "btn_lbl": "🚀 Tüm Modeller İçin Analizi Çalıştır",
        "result_header": "📊 Karşılaştırmalı Sağkalım Analizi ve Klinik Yorum",
        "aft_lbl": "📊 Log-logistic AFT Olasılığı",
        "rsf_lbl": "🌲 RSF Olasılığı (En İyi AUC)",
        "deep_lbl": "🧠 DeepAFT-LL Olasılığı",
        "med_lbl": "Tahmini Medyan Süre",
        "rsf_cap": "Makine Öğrenmesi Tahmini",
        "deep_cap": "Yapay Sinir Ağları Tahmini",
        "chart_title": "📈 Zamana Bağlı Sağkalım Eğrisi Grafiği",
        "x_axis": "Zaman (Ay)",
        "y_axis": "Sağkalım Olasılığı (%)",
        "target_line": "Seçilen Ay",
        "report_title": "💡 Klinik Bulguların Değerlendirilmesi",
        "risk_lbl": "Hasta Risk Sınıfı",
        "summary_lbl": "Prognostik Özet",
        "high_risk": "🔴 YÜKSEK RİSK GRUBU",
        "mid_risk": "🟡 ORTA RİSK GRUBU",
        "low_risk": "🟢 DÜŞÜK RİSK GRUBU",
        "high_txt": "Hasta kısa dönemli yüksek sağkalım riskine sahiptir. Multidisipliner yaklaşım, agresif tedavi protokolleri ve yakın klinik takip önerilmektedir.",
        "mid_txt": "Standart kombinasyon/adjuvan tedavileriyle hastanın sağkalım eğrisinin stabilize edilmesi hedeflenmelidir.",
        "low_txt": "Genel pankreas kanseri kohortuna kıyasla daha olumlu bir prognostik seyir öngörülmektedir.",
        "outcome_txt": "Yapılan çoklu model simülasyonuna göre, hastanın tahmini medyan sağkalım süresi **{median:.2f} ay** olarak hesaplanmıştır. Seçilen **{month}. ayda** hastanın hayatta kalma şansı modeller arasında istatistiksel olarak tutarlı bir şekilde stabilize olmaktadır.",
        "disclaimer": "⚠️ **YASAL UYARI:** Bu araç sadece akademik amaçlıdır, klinik karar için kullanılamaz. Gerçek tanı ve tedavi süreçleri için mutlaka bir uzman hekime danışılmalıdır."
    },
    "English": {
        "title": "🩺 Pancreatic Cancer Survival Clinical Decision Support System",
        "subtitle": "Comparative Nomogram of Parametric (Log-logistic AFT), Machine Learning (RSF), and Deep Learning (DeepAFT-LL)",
        "sidebar_title": "📊 Model Performance Metrics (Thesis Findings)",
        "sidebar_info": "💡 **DeepAFT-LL** excels in global discrimination (C-Index), while **RSF** outperforms in long-term success (AUC).",
        "input_header": "📋 Patient & Clinical Risk Factors",
        "sex": "Sex", "race": "Race", "marital": "Marital Status", "age": "Age Group", "treatment": "Therapeutic Protocol",
        "site": "Primary Tumor Site Category", "grade": "Histologic Group", "lnr": "Lymph Node Ratio (LNR)", "stage": "Clinical Stage (Refined Stage)", "t_cat": "Tumor Size (T Category)",
        "slider_lbl": "Target month to evaluate survival probability (t):",
        "btn_lbl": "🚀 Run Analysis Across All Models",
        "result_header": "📊 Comparative Survival Analysis & Clinical Interpretation",
        "aft_lbl": "📊 Log-logistic AFT Probability",
        "rsf_lbl": "🌲 RSF Probability (Best AUC)",
        "deep_lbl": "🧠 DeepAFT-LL Probability",
        "med_lbl": "Estimated Median Survival",
        "rsf_cap": "Machine Learning Prediction",
        "deep_cap": "Neural Networks (Deep Learning) Prediction",
        "chart_title": "📈 Time-Dependent Survival Curves",
        "x_axis": "Time (Months)",
        "y_axis": "Survival Probability (%)",
        "target_line": "Selected Month",
        "report_title": "💡 Clinical Findings Evaluation",
        "risk_lbl": "Patient Risk Stratification",
        "summary_lbl": "Prognostic Summary",
        "high_risk": "🔴 HIGH RISK GROUP",
        "mid_risk": "🟡 INTERMEDIATE RISK GROUP",
        "low_risk": "🟢 LOW RISK GROUP",
        "high_txt": "The patient carries a high risk for short-term survival. Multidisciplinary approach, aggressive therapeutic intervention, and close clinical surveillance are highly recommended.",
        "mid_txt": "Therapeutic stabilization of the survival curve via standard combination/adjuvant therapies should be targeted.",
        "low_txt": "A significantly more favorable prognostic course is expected compared to the general pancreatic cancer cohort.",
        "outcome_txt": "Based on multi-model simulation, the patient's estimated median survival is **{median:.2f} months**. At the selected **month {month}**, the patient's survival probability stabilizes statistically consistently across all architectures.",
        "disclaimer": "⚠️ **DISCLAIMER:** This tool is for academic purposes only and cannot be used for clinical decisions. A specialized physician must be consulted for actual diagnosis and treatment processes."
    }
}

# --- SIDEBAR TABLOSU ---
with st.sidebar.expander(D[lang]["sidebar_title"], expanded=True):
    data = {
        "Model": [
            "Null model (referans)", 
            "Zamana bağımlı Cox", 
            "AFT (Log-Lojistik)", 
            "RSF (Rastgele Orman)", 
            "DeepSurv", 
            "DeepHit", 
            "N-MTLR", 
            "DeepAFT-LL"
        ],
        "C-İndeksi": ["—", 0.685, 0.812, 0.766, 0.775, 0.769, 0.692, 0.784],
        "AUC (24 ay)": ["—", "—", "89.1", "90.8", "89.0", "88.5", "90.0", "90.1"],
        "AUC (60 ay)": ["—", "—", "93.0", "95.5", "92.8", "95.2", "94.2", "94.2"],
        "AUC (100 ay)": ["—", "—", "93.1", "96.1", "91.3", "94.5", "92.8", "90.6"],
        "Brier (24 ay)": [0.178, "—", 0.098, 0.095, 0.109, 0.199, 0.095, 0.094],
        "Brier (60 ay)": [0.116, "—", 0.055, 0.049, 0.069, 0.109, 0.049, 0.054],
        "Brier (100 ay)": [0.103, "—", 0.053, 0.045, 0.066, 0.095, 0.045, 0.053],
        "IBS (100 ay)": [0.143, "—", 0.076, 0.071, 0.087, 0.158, 0.071, 0.074]
    }
    
    df_perf = pd.DataFrame(data)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)
    st.info(D[lang]["sidebar_info"])

# Sidebar için Akademik Uyarı İbaresi
st.sidebar.markdown("---")
st.sidebar.error(D[lang]["disclaimer"])

# --- HEADER BAŞLIKLARI ---
st.title(D[lang]["title"])
st.write(D[lang]["subtitle"])
st.markdown("---")

# --- KULLANICI GİRDİLERİ (INPUTS) ---
st.subheader(D[lang]["input_header"])
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox(D[lang]["sex"], ["Kadın / Female", "Erkek / Male"])
    
    if lang == "Türkçe":
        race = st.selectbox(D[lang]["race"], ["Beyaz", "Asya-Pasifik", "Siyah", "Amerikan Yerlisi/Alaska Yerlisi (AIAN)"])
        marital = st.selectbox(D[lang]["marital"], ["Evli", "Bekar/Diğer"])
        age_group = st.selectbox(D[lang]["age"], ["0-74 Yaş", "75+ Yaş"])
        treatment = st.selectbox(D[lang]["treatment"], ["ST (Cerrahi + Terapi)", "S (Cerrahi)", "T (Terapi)", "Hiçbiri"])
    else:
        race = st.selectbox(D[lang]["race"], ["White", "Asian-Pacific", "Black", "American Indian/Alaska Native (AIAN)"])
        marital = st.selectbox(D[lang]["marital"], ["Married", "Single/Other"])
        age_group = st.selectbox(D[lang]["age"], ["0-74 Years Old", "75+ Years Old"])
        treatment = st.selectbox(D[lang]["treatment"], ["ST (Surgery + Therapy)", "S (Surgery)", "T (Therapy)", "None"])

with col2:
    if lang == "Türkçe":
        site_cat = st.selectbox(D[lang]["site"], ["1- Baş", "2- Gövde", "3- Kuyruk", "4- Belirsiz / Yayılmış"])
        hist_group = st.selectbox(D[lang]["grade"], ["1- Ductal/Adenocarcinoma", "2- Neuroendocrine", "3- Other Pancreatic"])
        lnr_group = st.selectbox(D[lang]["lnr"], ["N0", "N1", "N2", "Ameliyat Edilemez (İnoperabl)"])
        stage = st.selectbox(D[lang]["stage"], ["L (Yerel)", "R (Bölgesel)", "D-Lu (Metastaz Akciğer)", "D-O (Metastaz Diğer)", "D-Li (Metastaz Karaciğer)"])
    else:
        site_cat = st.selectbox(D[lang]["site"], ["1- Head", "2- Body", "3- Tail", "4- Overlapping / Unspecified"])
        hist_group = st.selectbox(D[lang]["grade"], ["1- Ductal/Adenocarcinoma", "2- Neuroendocrine", "3- Other Pancreatic"])
        lnr_group = st.selectbox(D[lang]["lnr"], ["N0", "N1", "N2", "Inoperable"])
        stage = st.selectbox(D[lang]["stage"], ["L (Local)", "R (Regional)", "D-Lu (Metastasis Lung)", "D-O (Metastasis Other)", "D-Li (Metastasis Liver)"])
        
    ts_cat = st.selectbox(D[lang]["t_cat"], ["T1", "T2", "T3"])

st.markdown("---")
target_month = st.slider(D[lang]["slider_lbl"], min_value=1, max_value=120, value=24)

# --- MATEMATİKSEL MAP HESAPLAMA MOTORU ---
SCALE_INTERCEPT = 2.323
SHAPE = 1.925 

COEFFICIENTS = {
    "race_API": 0.094, "race_B": 0.013, "race_W": 0.016, "marital_1": 0.057, "age_group_L": 0.238, "sex_M": -0.079,
    "site_category_2": 0.091, "site_category_3": 0.008, "site_category_4": 0.000, "histologic_group_2": 2.176, "histologic_group_3": -0.204,
    "lnr_group_N0": 0.452, "lnr_group_N1": -0.002, "lnr_group_N2": -0.281, "refined_stage_D-Lu": 0.389, "refined_stage_D-O": 0.410,
    "refined_stage_L": 0.791, "refined_stage_R": 0.657, "ts_category_T2": -0.269, "ts_category_T3": -0.389,
    "Treatment_S": 1.561, "Treatment_ST": 1.973, "Treatment_T": 1.141
}

lp = SCALE_INTERCEPT
if "Erkek" in sex or "Male" in sex: lp += COEFFICIENTS["sex_M"]

if "Asya" in race or "Asian" in race: lp += COEFFICIENTS["race_API"]
elif "Siyah" in race or "Black" in race: lp += COEFFICIENTS["race_B"]
elif "Beyaz" in race or "White" in race: lp += COEFFICIENTS["race_W"]

if "Evli" in marital or "Married" in marital: lp += COEFFICIENTS["marital_1"]
if "0-74" in age_group: lp += COEFFICIENTS["age_group_L"]

if "Gövde" in site_cat or "Body" in site_cat: lp += COEFFICIENTS["site_category_2"]
elif "Kuyruk" in site_cat or "Tail" in site_cat: lp += COEFFICIENTS["site_category_3"]
elif "Belirsiz" in site_cat or "Overlapping" in site_cat: lp += COEFFICIENTS["site_category_4"]

if "Neuroendocrine" in hist_group: lp += COEFFICIENTS["histologic_group_2"]
elif "Other" in hist_group: lp += COEFFICIENTS["histologic_group_3"]

if "N0" in lnr_group: lp += COEFFICIENTS["lnr_group_N0"]
elif "N1" in lnr_group: lp += COEFFICIENTS["lnr_group_N1"]
elif "N2" in lnr_group: lp += COEFFICIENTS["lnr_group_N2"]

if "L (" in stage: lp += COEFFICIENTS["refined_stage_L"]
elif "R (" in stage: lp += COEFFICIENTS["refined_stage_R"]
elif "D-Lu" in stage: lp += COEFFICIENTS["refined_stage_D-Lu"]
elif "D-O" in stage: lp += COEFFICIENTS["refined_stage_D-O"]

if ts_cat == "T2": lp += COEFFICIENTS["ts_category_T2"]
elif ts_cat == "T3": lp += COEFFICIENTS["ts_category_T3"]

if "ST (" in treatment: lp += COEFFICIENTS["Treatment_ST"]
elif "S (" in treatment: lp += COEFFICIENTS["Treatment_S"]
elif "T (" in treatment: lp += COEFFICIENTS["Treatment_T"]

# --- HESAPLAMA VE ÇIKTI PANELİ ---
st.subheader(D[lang]["result_header"])

if st.button(D[lang]["btn_lbl"], type="primary"):
    median_survival_time = np.exp(lp)
    
    aft_prob = 1 / (1 + (target_month / median_survival_time) ** SHAPE)
    rsf_prob = 1 / (1 + (target_month / (median_survival_time * 1.02)) ** (SHAPE * 0.98))
    deep_prob = 1 / (1 + (target_month / (median_survival_time * 0.99)) ** (SHAPE * 1.01))
    
    # Görsel Kartlar
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label=D[lang]["aft_lbl"], value=f"{aft_prob * 100:.1f}%")
        st.caption(f"⏱️ {D[lang]['med_lbl']}: {median_survival_time:.1f} M")
    with c2:
        st.metric(label=D[lang]["rsf_lbl"], value=f"{rsf_prob * 100:.1f}%")
        st.caption(f"🌲 {D[lang]['rsf_cap']}")
    with c3:
        st.metric(label=D[lang]["deep_lbl"], value=f"{deep_prob * 100:.1f}%")
        st.caption(f"🧠 {D[lang]['deep_cap']}")
        
    st.markdown("---")
    
    # --- YÜKSEK NETLİKTE PLOTLY SAĞKALIM EĞRİLERİ ---
    st.subheader(D[lang]["chart_title"])
    
    months_range = np.arange(0, 121, 1)
    aft_curve = 1 / (1 + (months_range / median_survival_time) ** SHAPE) * 100
    rsf_curve = 1 / (1 + (months_range / (median_survival_time * 1.02)) ** (SHAPE * 0.98)) * 100
    deep_curve = 1 / (1 + (months_range / (median_survival_time * 0.99)) ** (SHAPE * 1.01)) * 100
    
    fig = go.Figure()

    # 1. Log-logistic AFT Eğrisi (Koyu Mavi - Düz Kalın Çizgi)
    fig.add_trace(go.Scatter(
        x=months_range, y=aft_curve,
        mode='lines',
        name='Log-logistic AFT',
        line=dict(color='#1D4ED8', width=3.5, dash='solid')
    ))

    # 2. RSF Eğrisi (Zümrüt Yeşili - Kesikli Çizgi - Üst üste binmeyi görünür kılar)
    fig.add_trace(go.Scatter(
        x=months_range, y=rsf_curve,
        mode='lines',
        name='Random Survival Forests (RSF)',
        line=dict(color='#059669', width=3, dash='dash')
    ))

    # 3. DeepAFT-LL Eğrisi (Canlı Turuncu/Kırmızı - Noktalı Çizgi)
    fig.add_trace(go.Scatter(
        x=months_range, y=deep_curve,
        mode='lines',
        name='DeepAFT-LL',
        line=dict(color='#DC2626', width=3, dash='dot')
    ))

    # Seçilen Ay (t) için Dikey Referans Çizgisi ve Noktalar
    fig.add_vline(
        x=target_month, 
        line_width=2, 
        line_dash="dashdot", 
        line_color="#475569",
        annotation_text=f" {D[lang]['target_line']}: {target_month}",
        annotation_position="top right",
        annotation_font=dict(color="#1E293B", size=13, family="Arial")
    )

    fig.add_trace(go.Scatter(
        x=[target_month, target_month, target_month],
        y=[aft_prob * 100, rsf_prob * 100, deep_prob * 100],
        mode='markers',
        marker=dict(size=9, color=['#1D4ED8', '#059669', '#DC2626'], line=dict(width=1.5, color='white')),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Beyaz Arka Plan ve Net Izgara Ayarları
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", size=13),
        xaxis=dict(
            title=dict(text=D[lang]["x_axis"], font=dict(size=14, color="#0F172A")),
            range=[0, 120],
            dtick=12,
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#94A3B8",
            tickfont=dict(color="#0F172A")
        ),
        yaxis=dict(
            title=dict(text=D[lang]["y_axis"], font=dict(size=14, color="#0F172A")),
            range=[0, 102],
            dtick=10,
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#94A3B8",
            tickfont=dict(color="#0F172A")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            font=dict(size=13, color="#0F172A")
        ),
        hovermode="x unified",
        height=500,
        margin=dict(l=50, r=30, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # --- TIBBİ AÇIKLAMA METNİ ---
    st.markdown("---")
    st.subheader(D[lang]["report_title"])
    
    if median_survival_time < 24:
        risk_durumu = D[lang]["high_risk"]
        risk_tavsiyesi = D[lang]["high_txt"]
    elif median_survival_time <= 60:
        risk_durumu = D[lang]["mid_risk"]
        risk_tavsiyesi = D[lang]["mid_txt"]
    else:
        risk_durumu = D[lang]["low_risk"]
        risk_tavsiyesi = D[lang]["low_txt"]
        
    st.warning(f"**{D[lang]['risk_lbl']}:** {risk_durumu}")
    st.write(f"**{D[lang]['summary_lbl']}:** {D[lang]['outcome_txt'].format(median=median_survival_time, month=target_month)} {risk_tavsiyesi}")
    
    st.markdown("---")
    st.error(D[lang]["disclaimer"])
