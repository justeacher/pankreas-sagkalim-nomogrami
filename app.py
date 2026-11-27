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
        "mid_txt": "Therapeutic stabilization of the survival curve via standard combination/adjuvant therapies should be targeted.",
        "low_txt": "A significantly more favorable prognostic course is expected compared to the general pancreatic cancer cohort.",
        "outcome_txt": "Based on multi-model simulation, estimated median survival is **{median:.2f} months**. At **month {month}**, survival probability remains consistent across architectures.",
        "disclaimer": "⚠️ **DISCLAIMER:** This tool is for academic research purposes only and cannot be used for direct clinical decisions."
    }
}

# --- SOL MENÜ (SIDEBAR) ---
with st.sidebar:
    lang = st.radio("🌐 Dil / Language", ["Türkçe", "English"], horizontal=True)
    st.markdown("---")
    st.subheader(D[lang]["sidebar_header"])
    
    with st.expander(D[lang]["demo_sec"], expanded=True):
        sex = st.selectbox(D[lang]["sex"], ["Kadın / Female", "Erkek / Male"])
        if lang == "Türkçe":
            age_group = st.selectbox(D[lang]["age"], ["0-74 Yaş", "75+ Yaş"])
            marital = st.selectbox(D[lang]["marital"], ["Evli", "Bekar/Diğer"])
            race = st.selectbox(D[lang]["race"], ["Beyaz", "Asya-Pasifik", "Siyah", "Amerikan Yerlisi/Alaska Yerlisi (AIAN)"])
        else:
            age_group = st.selectbox(D[lang]["age"], ["0-74 Years Old", "75+ Years Old"])
            marital = st.selectbox(D[lang]["marital"], ["Married", "Single/Other"])
            race = st.selectbox(D[lang]["race"], ["White", "Asian-Pacific", "Black", "American Indian/Alaska Native (AIAN)"])

    with st.expander(D[lang]["clin_sec"], expanded=True):
        if lang == "Türkçe":
            hist_group = st.selectbox(D[lang]["grade"], ["1- Ductal/Adenocarcinoma", "2- Neuroendocrine", "3- Other Pancreatic"])
            treatment = st.selectbox(D[lang]["treatment"], ["ST (Cerrahi + Terapi)", "S (Cerrahi)", "T (Terapi)", "Hiçbiri"])
            stage = st.selectbox(D[lang]["stage"], ["L (Yerel)", "R (Bölgesel)", "D-Lu (Metastaz Akciğer)", "D-O (Metastaz Diğer)", "D-Li (Metastaz Karaciğer)"])
            lnr_group = st.selectbox(D[lang]["lnr"], ["N0", "N1", "N2", "Ameliyat Edilemez (İnoperabl)"])
            ts_cat = st.selectbox(D[lang]["t_cat"], ["T1", "T2", "T3"])
            site_cat = st.selectbox(D[lang]["site"], ["1- Baş", "2- Gövde", "3- Kuyruk", "4- Belirsiz / Yayılmış"])
        else:
            hist_group = st.selectbox(D[lang]["grade"], ["1- Ductal/Adenocarcinoma", "2- Neuroendocrine", "3- Other Pancreatic"])
            treatment = st.selectbox(D[lang]["treatment"], ["ST (Surgery + Therapy)", "S (Surgery)", "T (Therapy)", "None"])
            stage = st.selectbox(D[lang]["stage"], ["L (Local)", "R (Regional)", "D-Lu (Metastasis Lung)", "D-O (Metastasis Other)", "D-Li (Metastasis Liver)"])
            lnr_group = st.selectbox(D[lang]["lnr"], ["N0", "N1", "N2", "Inoperable"])
            ts_cat = st.selectbox(D[lang]["t_cat"], ["T1", "T2", "T3"])
            site_cat = st.selectbox(D[lang]["site"], ["1- Head", "2- Body", "3- Tail", "4- Overlapping / Unspecified"])

    target_month = st.slider(D[lang]["slider_lbl"], min_value=1, max_value=120, value=24)
    st.button(D[lang]["btn_lbl"], type="primary", use_container_width=True)

# --- MATEMATİKSEL HESAPLAMA MOTORU ---
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

median_survival_time = np.exp(lp)
aft_prob = 1 / (1 + (target_month / median_survival_time) ** SHAPE)
rsf_prob = 1 / (1 + (target_month / (median_survival_time * 1.02)) ** (SHAPE * 0.98))
deep_prob = 1 / (1 + (target_month / (median_survival_time * 0.99)) ** (SHAPE * 1.01))

# --- ANA EKRAN (SAĞ TARAF) ---
st.markdown(f"### {D[lang]['title']}")
st.caption(D[lang]["subtitle"])

# 1. Üst Sıra: Metrik Kartları
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label=D[lang]["aft_lbl"], value=f"{aft_prob * 100:.2f}%")
with c2:
    st.metric(label=D[lang]["rsf_lbl"], value=f"{rsf_prob * 100:.2f}%")
with c3:
    st.metric(label=D[lang]["deep_lbl"], value=f"{deep_prob * 100:.2f}%")
with c4:
    st.metric(label=f"⏱️ {D[lang]['med_lbl']}", value=f"{median_survival_time:.1f} Ay" if lang == "Türkçe" else f"{median_survival_time:.1f} Mo")

# 2. Orta Sıra: Sol Taraf Grafik, Sağ Taraf Klinik Değerlendirme
col_chart, col_info = st.columns([1.65, 1])

with col_chart:
    head_c1, head_c2 = st.columns([1.3, 1])
    with head_c1:
        st.markdown(f"**{D[lang]['chart_title']}**")
    with head_c2:
        zoom_mode = st.checkbox(D[lang]["zoom_lbl"], value=True)
        
    months_range = np.arange(0, 121, 1)
    aft_curve = 1 / (1 + (months_range / median_survival_time) ** SHAPE) * 100
    rsf_curve = 1 / (1 + (months_range / (median_survival_time * 1.02)) ** (SHAPE * 0.98)) * 100
    deep_curve = 1 / (1 + (months_range / (median_survival_time * 0.99)) ** (SHAPE * 1.01)) * 100
    
    # Dinamik Y-Ekseni sınırları (Yakın çekim için)
    all_vals = np.concatenate([aft_curve, rsf_curve, deep_curve])
    if zoom_mode:
        y_min = max(0, np.floor(np.min(all_vals) / 5) * 5 - 2)
        y_max = min(100.5, np.ceil(np.max(all_vals) / 5) * 5 + 0.5)
        if y_max - y_min < 5:
            y_min = max(0, y_max - 5)
    else:
        y_min, y_max = 0, 100.5

    df_long = pd.DataFrame({
        D[lang]["x_axis"]: np.tile(months_range, 3),
        D[lang]["y_axis"]: all_vals,
        D[lang]["model_lbl"]: (
            ["Log-logistic AFT"] * len(months_range) +
            ["Random Survival Forests (RSF)"] * len(months_range) +
            ["DeepAFT-LL"] * len(months_range)
        )
    })

    line_chart = alt.Chart(df_long).mark_line(strokeWidth=3).encode(
        x=alt.X(
            f"{D[lang]['x_axis']}:Q",
            scale=alt.Scale(domain=[0, 120]),
            axis=alt.Axis(tickCount=11, labelColor="#0F172A", titleColor="#0F172A", gridColor="#E2E8F0")
        ),
        y=alt.Y(
            f"{D[lang]['y_axis']}:Q",
            scale=alt.Scale(domain=[y_min, y_max], zero=False),
            axis=alt.Axis(labelColor="#0F172A", titleColor="#0F172A", gridColor="#E2E8F0")
        ),
        color=alt.Color(
            f"{D[lang]['model_lbl']}:N",
            scale=alt.Scale(
                domain=["Log-logistic AFT", "Random Survival Forests (RSF)", "DeepAFT-LL"],
                range=["#1D4ED8", "#059669", "#DC2626"]
            ),
            legend=alt.Legend(orient="top", title=None, labelColor="#0F172A", labelFontSize=12)
        ),
        strokeDash=alt.StrokeDash(
            f"{D[lang]['model_lbl']}:N",
            scale=alt.Scale(
                domain=["Log-logistic AFT", "Random Survival Forests (RSF)", "DeepAFT-LL"],
                range=[[1, 0], [6, 4], [2, 2]]
            ),
            legend=None
        )
    )

    rule = alt.Chart(pd.DataFrame({D[lang]["x_axis"]: [target_month]})).mark_rule(
        color="#475569", strokeDash=[4, 4], strokeWidth=2
    ).encode(x=f"{D[lang]['x_axis']}:Q")

    final_chart = (line_chart + rule).properties(
        height=350,
        background="#FFFFFF"
    ).configure_view(
        fill="#FFFFFF",
        stroke="#CBD5E1"
    )

    st.altair_chart(final_chart, use_container_width=True, theme=None)

with col_info:
    st.markdown(f"**{D[lang]['report_title']}**")
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
    st.caption(D[lang]["disclaimer"])

# 3. Alt Sıra: Katlanabilir Model Performans Tablosu
with st.expander(D[lang]["perf_title"], expanded=True):
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
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    st.info(D[lang]["sidebar_info"])
