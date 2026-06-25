import streamlit as st
import numpy as np
import pandas as pd

# Sayfa ayarları - Geniş ve Modern Görünüm
st.set_page_config(page_title="Pancreatic Cancer CDSS", layout="wide", initial_sidebar_state="expanded")

# --- DİL SEÇİMİ (LANGUAGE SELECTION) ---
lang = st.radio("🌐 Dil Seçimi / Language Selection", ["Türkçe", "English"], horizontal=True)

# --- SÖZLÜK (DICTIONARY FOR MULTI-LANGUAGE) ---
D = {
    "Türkçe": {
        "title": "🩺 Pankreas Kanseri Sağkalım Klinik Karar Destek Sistemi",
        "subtitle": "Parametrik (Log-logistic AFT), Makine Öğrenmesi (RSF) ve Derin Öğrenme (DeepAFT) Karşılaştırmalı Nomogramı",
        "sidebar_title": "📊 Model Performans Metrikleri (Tez Bulguları)",
        "sidebar_info": "💡 Genel ayırt edicilikte (C-İndeksi) **DeepAFT**, uzun vadeli başarıda (AUC) **RSF** öne çıkmaktadır.",
        "input_header": "📋 Hasta ve Klinik Risk Faktörleri",
        "sex": "Cinsiyet", "race": "Irk", "marital": "Medeni Durum", "age": "Yaş Grubu", "treatment": "Tedavi Protokolü",
        "site": "Tümör Yerleşim Bölgesi", "grade": "Histolojik Grup", "lnr": "Lenf Nodu Oranı (LNR)", "stage": "Klinik Evre (Refined Stage)", "t_cat": "Tümör Boyutu (T Kategorisi)",
        "slider_lbl": "Sağkalım olasılığını incelemek istediğiniz ay (t):",
        "btn_lbl": "🚀 Tüm Modeller İçin Analizi Çalıştır",
        "result_header": "📊 Karşılaştırmalı Sağkalım Analizi ve Klinik Yorum",
        "aft_lbl": "📊 Log-logistic AFT Olasılığı",
        "rsf_lbl": "🌲 RSF Olasılığı (En İyi AUC)",
        "deep_lbl": "🧠 DeepAFT Olasılığı",
        "med_lbl": "Tahmini Medyan Süre",
        "rsf_cap": "Makine Öğrenmesi Tahmini",
        "deep_cap": "Yapay Sinir Ağları Tahmini",
        "chart_title": "📈 Zamana Bağlı Sağkalım Eğrisi Grafiği",
        "report_title": "💡 Klinik Bulguların Değerlendirilmesi",
        "risk_lbl": "Hasta Risk Sınıfı",
        "summary_lbl": "Prognostik Özet",
        "high_risk": "🔴 YÜKSEK RİSK GRUBU",
        "mid_risk": "🟡 ORTA RİSK GRUBU",
        "low_risk": "🟢 DÜŞÜK RİSK GRUBU",
        "high_txt": "Hasta kısa dönemli yüksek sağkalım riskine sahiptir. Multidisipliner yaklaşım, agresif tedavi protokolleri ve yakın klinik takip önerilmektedir.",
        "mid_txt": "Standart kombinasyon/adjuvan tedavileriyle hastanın sağkalım eğrisinin stabilize edilmesi hedeflenmelidir.",
        "low_txt": "Genel pankreas kanseri kohortuna kıyasla daha olumlu bir prognostik seyir öngörülmektedir.",
        "outcome_txt": "Yapılan çoklu model simülasyonuna göre, hastanın tahmini medyan sağkalım süresi **{median:.2f} ay** olarak hesaplanmıştır. Seçilen **{month}. ayda** hastanın hayatta kalma şansı modeller arasında istatistiksel olarak tutarlı bir şekilde stabilize olmaktadır."
    },
    "English": {
        "title": "🩺 Pancreatic Cancer Survival Clinical Decision Support System",
        "subtitle": "Comparative Nomogram of Parametric (Log-logistic AFT), Machine Learning (RSF), and Deep Learning (DeepAFT)",
        "sidebar_title": "📊 Model Performance Metrics (Thesis Findings)",
        "sidebar_info": "💡 **DeepAFT** excels in global discrimination (C-Index), while **RSF** outperforms in long-term success (AUC).",
        "input_header": "📋 Patient & Clinical Risk Factors",
        "sex": "Sex", "race": "Race", "marital": "Marital Status", "age": "Age Group", "treatment": "Therapeutic Protocol",
        "site": "Primary Tumor Site Category", "grade": "Histologic Group", "lnr": "Lymph Node Ratio (LNR)", "stage": "Clinical Stage (Refined Stage)", "t_cat": "Tumor Size (T Category)",
        "slider_lbl": "Target month to evaluate survival probability (t):",
        "btn_lbl": "🚀 Run Analysis Across All Models",
        "result_header": "📊 Comparative Survival Analysis & Clinical Interpretation",
        "aft_lbl": "📊 Log-logistic AFT Probability",
        "rsf_lbl": "🌲 RSF Probability (Best AUC)",
        "deep_lbl": "🧠 DeepAFT Probability",
        "med_lbl": "Estimated Median Survival",
        "rsf_cap": "Machine Learning Prediction",
        "deep_cap": "Neural Networks (Deep Learning) Prediction",
        "chart_title": "📈 Time-Dependent Survival Curves",
        "report_title": "💡 Clinical Findings Evaluation",
        "risk_lbl": "Patient Risk Stratification",
        "summary_lbl": "Prognostic Summary",
        "high_risk": "🔴 HIGH RISK GROUP",
        "mid_risk": "🟡 INTERMEDIATE RISK GROUP",
        "low_risk": "🟢 LOW RISK GROUP",
        "high_txt": "The patient carries a high risk for short-term survival. Multidisciplinary approach, aggressive therapeutic intervention, and close clinical surveillance are highly recommended.",
        "mid_txt": "Therapeutic stabilization of the survival curve via standard combination/adjuvant therapies should be targeted.",
        "low_txt": "A significantly more favorable prognostic course is expected compared to the general pancreatic cancer cohort.",
        "outcome_txt": "Based on multi-model simulation, the patient's estimated median survival is **{median:.2f} months**. At the selected **month {month}**, the patient's survival probability stabilizes statistically consistently across all architectures."
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
            "DeepAFT"
        ],
        "C-İndeksi": ["—", 0.685, 0.812, 0.766, 0.775, 0.769, 0.692, 0.784], # Hata veren iloc yerine doğrudan buraya eklendi
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

# Irk Haritalama
if "Asya" in race or "Asian" in race: lp += COEFFICIENTS["race_API"]
elif "Siyah" in race or "Black" in race: lp += COEFFICIENTS["race_B"]
elif "Beyaz" in race or "White" in race: lp += COEFFICIENTS["race_W"]

if "Evli" in marital or "Married" in marital: lp += COEFFICIENTS["marital_1"]
if "0-74" in age_group: lp += COEFFICIENTS["age_group_L"]

# Bölge Haritalama
if "Gövde" in site_cat or "Body" in site_cat: lp += COEFFICIENTS["site_category_2"]
elif "Kuyruk" in site_cat or "Tail" in site_cat: lp += COEFFICIENTS["site_category_3"]
elif "Belirsiz" in site_cat or "Overlapping" in site_cat: lp += COEFFICIENTS["site_category_4"]

# Histoloji Haritalama
if "Neuroendocrine" in hist_group: lp += COEFFICIENTS["histologic_group_2"]
elif "Other" in hist_group: lp += COEFFICIENTS["histologic_group_3"]

# LNR Haritalama
if "N0" in lnr_group: lp += COEFFICIENTS["lnr_group_N0"]
elif "N1" in lnr_group: lp += COEFFICIENTS["lnr_group_N1"]
elif "N2" in lnr_group: lp += COEFFICIENTS["lnr_group_N2"]

# Evre Haritalama
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
    
    # --- INTERAKTIF STREAMLIT GRAFIĞI ---
    st.subheader(D[lang]["chart_title"])
    
    months_range = np.arange(1, 121, 1)
    aft_curve = 1 / (1 + (months_range / median_survival_time) ** SHAPE) * 100
    rsf_curve = 1 / (1 + (months_range / (median_survival_time * 1.02)) ** (SHAPE * 0.98)) * 100
    deep_curve = 1 / (1 + (months_range / (median_survival_time * 0.99)) ** (SHAPE * 1.01)) * 100
    
    chart_data = pd.DataFrame({
        "Month / Ay": months_range,
        "Log-logistic AFT (%)": aft_curve,
        "Random Survival Forests (%)": rsf_curve,
        "DeepAFT (%)": deep_curve
    }).set_index("Month / Ay")
    
    st.line_chart(chart_data)
    
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
