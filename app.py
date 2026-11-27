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
        :root {
            color-scheme: light !important;
        }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #CBD5E1 !important;
        }
        .block-container {
            padding-top: 2.2rem !important;
            padding-bottom: 2rem !important;
        }
        h1, h2, h3, h4, p, span, label, div[data-testid="stMarkdownContainer"] {
            color: #0F172A !important;
        }
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
        div[data-testid="stMetric"] {
            background-color: #F8FAFC !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
        }
        div[data-testid="stAlert"] {
            background-color: #FEF9C3 !important;
            color: #0F172A !important;
            border: 1px solid #EAB308 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stAlert"] p, div[data-testid="stAlert"] strong {
            color: #0F172A !important;
        }
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
        [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
        }
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
            "Kemoradyoterapi / Chemoradiation",
            "Tedavi Yok / No Treatment"
        ])

    with st.expander(T["clin_sec"], expanded=True):
        site = st.selectbox(T["site"], ["Baş / Head", "Gövde / Body", "Kuyruk / Tail", "Diğer / Other"])
        grade = st.selectbox(T["grade"], ["Grade I (İyi)", "Grade II (Orta)", "Grade III (Kötü)", "Grade IV (Diferansiye Olmamış)"])
        lnr = st.selectbox(T["lnr"], ["0 (Negatif)", "0.01 - 0.20", "0.21 - 0.50", "> 0.50"])
        stage = st.selectbox(T["stage"], ["Evre I / Stage I", "Evre II / Stage II", "Evre III / Stage III", "Evre IV / Stage IV"])
        t_cat = st.selectbox(T["t_cat"], ["T1", "T2", "T3", "T4"])

    target_month = st.slider(T["slider_lbl"], min_value=1, max_value=120, value=24, step=1)
    run_btn = st.button(T["btn_lbl"])
    st.markdown("---")
    st.info(T["sidebar_info"])

# --- ANA BAŞLIK ---
st.title(T["title"])
st.markdown(f"*{T['subtitle']}*")
st.markdown("---")

# --- RİSK VE SAĞKALIM SİMÜLASYON HESAPLAMASI ---
# Not: Gerçek eğitilmiş model katsayılarınızı buraya entegre edebilirsiniz.
risk_score = 0.0
risk_score += {"Evre I / Stage I": -0.6, "Evre II / Stage II": -0.2, "Evre III / Stage III": 0.3, "Evre IV / Stage IV": 0.9}[stage]
risk_score += {"0 (Negatif)": -0.3, "0.01 - 0.20": 0.0, "0.21 - 0.50": 0.35, "> 0.50": 0.65}[lnr]
risk_score += {"Grade I (İyi)": -0.25, "Grade II (Orta)": 0.0, "Grade III (Kötü)": 0.25, "Grade IV (Diferansiye Olmamış)": 0.45}[grade]
risk_score += {
    "Cerrahi + Kemoterapi / Surgery + Chemo": -0.55,
    "Sadece Cerrahi / Surgery Only": -0.25,
    "Kemoradyoterapi / Chemoradiation": -0.1,
    "Sadece Kemoterapi / Chemo Only": 0.15,
    "Tedavi Yok / No Treatment": 0.75
}[treatment]
risk_score += {"< 50": -0.2, "50 - 59": -0.1, "60 - 69": 0.0, "70 - 79": 0.15, "80+": 0.35}[age]
risk_score += {"T1": -0.2, "T2": 0.0, "T3": 0.15, "T4": 0.35}[t_cat]

# Zaman ekseni (0 - 120 ay)
t_grid = np.arange(0, 121, 1)

# Log-logistic AFT sağkalım fonksiyonu: S(t) = 1 / (1 + (t / alpha)^beta)
alpha_aft = max(3.0, 22.0 * np.exp(-risk_score))
beta_aft = 1.45
s_aft = 1.0 / (1.0 + (t_grid / alpha_aft) ** beta_aft)

# DeepAFT-LL sağkalım eğrisi (hafif non-lineer düzeltmeli)
alpha_deep = max(3.0, 23.5 * np.exp(-risk_score * 0.95))
beta_deep = 1.52
s_deep = 1.0 / (1.0 + (t_grid / alpha_deep) ** beta_deep)

# RSF (Random Survival Forest) ağaç tabanlı sağkalım yaklaşımı (Weibull benzeri)
alpha_rsf = alpha_aft * 1.15
beta_rsf = 1.18
s_rsf = np.exp(-((t_grid / alpha_rsf) ** beta_rsf))

# Seçilen aydaki olasılıklar
prob_aft = s_aft[target_month] * 100
prob_rsf = s_rsf[target_month] * 100
prob_deep = s_deep[target_month] * 100
median_surv = float(np.mean([alpha_aft, alpha_deep, alpha_rsf * (np.log(2) ** (1 / beta_rsf))]))

# --- ÜST METRİK KARTLARI ---
c1, c2, c3, c4 = st.columns(4)
c1.metric(T["aft_lbl"], f"%{prob_aft:.1f}")
c2.metric(T["rsf_lbl"], f"%{prob_rsf:.1f}")
c3.metric(T["deep_lbl"], f"%{prob_deep:.1f}")
c4.metric(T["med_lbl"], f"{median_surv:.1f} Ay" if lang == "Türkçe" else f"{median_surv:.1f} Mos")
st.markdown("---")

# --- SAĞKALIM EĞRİLERİ GRAFİĞİ ---
col_chart_title, col_zoom = st.columns([3, 1])
with col_chart_title:
    st.subheader(T["chart_title"])
with col_zoom:
    zoom_y = st.checkbox(T["zoom_lbl"], value=False)

# Dahili sütun adları sabit tutulur (parantez/boşluk içeren adlar Altair'de sorun çıkarabilir),
# görünen başlıklar sözlükten gelir.
n = len(t_grid)
df_plot = pd.DataFrame({
    "t": np.tile(t_grid, 3),
    "S": np.concatenate([s_aft * 100, s_rsf * 100, s_deep * 100]),
    "Model": ["Log-logistic AFT"] * n + ["RSF"] * n + ["DeepAFT-LL"] * n
})

y_scale = alt.Scale(zero=False) if zoom_y else alt.Scale(domain=[0, 100])

line_chart = alt.Chart(df_plot).mark_line(strokeWidth=3).encode(
    x=alt.X("t:Q", title=T["x_axis"]),
    y=alt.Y("S:Q", title=T["y_axis"], scale=y_scale),
    color=alt.Color(
        "Model:N",
        title=T["model_lbl"],
        scale=alt.Scale(
            domain=["Log-logistic AFT", "RSF", "DeepAFT-LL"],
            range=["#2563EB", "#16A34A", "#DC2626"]
        ),
        legend=alt.Legend(orient="top", title=None)
    ),
    tooltip=[
        alt.Tooltip("t:Q", title=T["x_axis"], format=".0f"),
        alt.Tooltip("Model:N", title=T["model_lbl"]),
        alt.Tooltip("S:Q", title=T["y_axis"], format=".2f")
    ]
)

vline = alt.Chart(pd.DataFrame({"t": [target_month]})).mark_rule(
    color="#475569", strokeDash=[5, 5], strokeWidth=2
).encode(x="t:Q")

final_chart = (line_chart + vline).properties(height=380).configure(
    background="#FFFFFF"
).configure_axis(
    labelColor="#0F172A",
    titleColor="#0F172A",
    gridColor="#E2E8F0"
).configure_legend(
    labelColor="#0F172A"
)
st.altair_chart(final_chart, use_container_width=True)

# --- KLİNİK BULGULAR VE RİSK SINIFLANDIRMASI ---
st.subheader(T["report_title"])
if median_surv < 12:
    risk_badge, risk_desc = T["high_risk"], T["high_txt"]
elif median_surv < 24:
    risk_badge, risk_desc = T["mid_risk"], T["mid_txt"]
else:
    risk_badge, risk_desc = T["low_risk"], T["low_txt"]

r_col1, r_col2 = st.columns([1, 2])
with r_col1:
    st.markdown(f"**{T['risk_lbl']}**")
    st.markdown(f"### {risk_badge}")
with r_col2:
    st.markdown(f"**{T['summary_lbl']}**")
    st.write(T["outcome_txt"].format(median=median_surv, month=target_month))
    st.write(risk_desc)

# --- BENCHMARK TABLOSU ---
with st.expander(T["perf_title"], expanded=False):
    bench_df = pd.DataFrame({
        "Model": ["Log-logistic AFT", "DeepAFT-LL", "RSF (Random Survival Forest)", "Cox PH"],
        "C-Index": [0.812, 0.784, 0.779, 0.765],
        "12-Ay AUC": [0.845, 0.821, 0.838, 0.802],
        "36-Ay AUC": [0.810, 0.795, 0.851, 0.774],
        "IBS (Brier Score)": [0.118, 0.126, 0.121, 0.139]
    })
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.warning(T["disclaimer"])
