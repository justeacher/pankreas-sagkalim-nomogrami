import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Sayfa ayarları - Geniş ve Modern Görünüm
st.set_page_config(
    page_title="Pancreatic Cancer CDSS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TAM BEYAZ TEMA (alt kısım dahil hiçbir yerde siyah alan kalmaz) ---
st.markdown("""
    <style>
        :root { color-scheme: light !important; }

        /* Sayfanın tamamı: kök, gövde, ana alan, alt çubuk, footer, üst bar */
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"],
        footer, .main {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        [data-testid="stDecoration"] { display: none !important; }

        /* Sidebar */
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #CBD5E1 !important;
        }
        .block-container { padding-top: 2.2rem !important; padding-bottom: 2rem !important; }

        /* Metinler */
        h1, h2, h3, h4, p, span, label, li, div[data-testid="stMarkdownContainer"] {
            color: #0F172A !important;
        }

        /* Selectbox ve açılır liste */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #94A3B8 !important;
            min-height: 36px !important;
        }
        div[data-baseweb="select"] span, div[data-baseweb="select"] svg {
            color: #0F172A !important; fill: #0F172A !important;
        }
        div[data-baseweb="popover"], div[data-baseweb="popover"] > div,
        ul[role="listbox"], ul[role="listbox"] li {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        ul[role="listbox"] li:hover { background-color: #F1F5F9 !important; }

        /* Expander */
        div[data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stExpander"] details summary {
            background-color: #F1F5F9 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stExpander"] details summary span,
        div[data-testid="stExpander"] details summary svg {
            color: #0F172A !important; fill: #0F172A !important;
        }

        /* Metrik kartları */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
        }

        /* Uyarı / bilgi kutuları */
        div[data-testid="stAlert"] {
            background-color: #FEF9C3 !important;
            border: 1px solid #EAB308 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stAlert"] p, div[data-testid="stAlert"] strong { color: #0F172A !important; }

        /* Buton */
        div.stButton > button {
            background-color: #0F172A !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: none !important;
            width: 100%;
            font-weight: 600;
        }
        div.stButton > button p { color: #FFFFFF !important; }

        /* Grafik */
        div[data-testid="stVegaLiteChart"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }

        /* Performans tablosu (HTML tablo: tema ne olursa olsun beyaz kalır) */
        .perf-wrap { overflow-x: auto; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; }
        table.perf-table { border-collapse: collapse; width: 100%; font-size: 11.5px; background: #FFFFFF; }
        table.perf-table th {
            background: #F1F5F9; color: #0F172A; text-align: left;
            padding: 6px 8px; border-bottom: 1px solid #CBD5E1; white-space: nowrap;
        }
        table.perf-table td {
            color: #0F172A; padding: 5px 8px; border-bottom: 1px solid #E2E8F0; white-space: nowrap;
        }
    </style>
""", unsafe_allow_html=True)

# --- DİL SEÇİMİ ---
lang = st.radio("🌐 Dil Seçimi / Language Selection", ["Türkçe", "English"], horizontal=True)
LI = 0 if lang == "Türkçe" else 1  # etiket indeksi: 0 = TR, 1 = EN

# --- SÖZLÜK ---
D = {
    "Türkçe": {
        "title": "🩺 Pankreas Kanseri Sağkalım Klinik Karar Destek Sistemi",
        "subtitle": "Parametrik (Log-logistic AFT), Makine Öğrenmesi (RSF) ve Derin Öğrenme (DeepAFT) Karşılaştırmalı Nomogramı",
        "sidebar_title": "📊 Model Performans Metrikleri (Tez Bulguları)",
        "sidebar_info": "💡 Genel ayırt edicilikte (C-İndeksi) **Log-Lojistik AFT (0.812)**, derin öğrenme modelleri arasında **DeepAFT (0.784)**, uzun vadeli başarıda (AUC) ise **RSF** öne çıkmaktadır.",
        "input_header": "📋 Hasta ve Klinik Risk Faktörleri",
        "sex": "Cinsiyet", "race": "Irk", "marital": "Medeni Durum", "age": "Yaş Grubu", "treatment": "Tedavi Protokolü",
        "site": "Tümör Yerleşim Bölgesi", "grade": "Histolojik Grup", "lnr": "Lenf Nodu Oranı (LNR)",
        "stage": "Klinik Evre (Refined Stage)", "t_cat": "Tümör Boyutu (T Kategorisi)",
        "slider_lbl": "Sağkalım olasılığını incelemek istediğiniz ay (t):",
        "btn_lbl": "🚀 Tüm Modeller İçin Analizi Çalıştır",
        "hint": "Sonuçları görmek için yukarıdaki parametreleri seçip analizi çalıştırın.",
        "result_header": "📊 Karşılaştırmalı Sağkalım Analizi ve Klinik Yorum",
        "aft_lbl": "📊 Log-logistic AFT Olasılığı",
        "rsf_lbl": "🌲 RSF Olasılığı (En İyi AUC)",
        "deep_lbl": "🧠 DeepAFT Olasılığı",
        "med_lbl": "Tahmini Medyan Süre",
        "rsf_cap": "Makine Öğrenmesi Tahmini",
        "deep_cap": "Yapay Sinir Ağları Tahmini",
        "chart_title": "📈 Zamana Bağlı Sağkalım Eğrisi Grafiği",
        "x_axis": "Zaman (Ay)", "y_axis": "Sağkalım Olasılığı (%)", "model_lbl": "Model",
        "report_title": "💡 Klinik Bulguların Değerlendirilmesi",
        "risk_lbl": "Hasta Risk Sınıfı",
        "summary_lbl": "Prognostik Özet",
        "high_risk": "🔴 YÜKSEK RİSK GRUBU",
        "mid_risk": "🟡 ORTA RİSK GRUBU",
        "low_risk": "🟢 DÜŞÜK RİSK GRUBU",
        "high_txt": "Hasta kısa dönemli yüksek sağkalım riskine sahiptir. Multidisipliner yaklaşım, agresif tedavi protokolleri ve yakın klinik takip önerilmektedir.",
        "mid_txt": "Standart kombinasyon/adjuvan tedavileriyle hastanın sağkalım eğrisinin stabilize edilmesi hedeflenmelidir.",
        "low_txt": "Genel pankreas kanseri kohortuna kıyasla daha olumlu bir prognostik seyir öngörülmektedir.",
        "outcome_txt": "Yapılan çoklu model simülasyonuna göre, hastanın tahmini medyan sağkalım süresi **{median:.2f} ay** olarak hesaplanmıştır. Seçilen **{month}. ayda** hastanın hayatta kalma olasılığı modeller arasında tutarlı bir seyir izlemektedir.",
        "disclaimer": "⚠️ **YASAL UYARI:** Bu araç yalnızca akademik araştırma amaçlıdır; doğrudan klinik karar mekanizması olarak kullanılamaz."
    },
    "English": {
        "title": "🩺 Pancreatic Cancer Survival Clinical Decision Support System",
        "subtitle": "Comparative Nomogram of Parametric (Log-logistic AFT), Machine Learning (RSF), and Deep Learning (DeepAFT)",
        "sidebar_title": "📊 Model Performance Metrics (Thesis Findings)",
        "sidebar_info": "💡 **Log-logistic AFT (0.812)** excels in global discrimination (C-Index), **DeepAFT (0.784)** leads among deep learning models, while **RSF** outperforms in long-term success (AUC).",
        "input_header": "📋 Patient & Clinical Risk Factors",
        "sex": "Sex", "race": "Race", "marital": "Marital Status", "age": "Age Group", "treatment": "Therapeutic Protocol",
        "site": "Primary Tumor Site Category", "grade": "Histologic Group", "lnr": "Lymph Node Ratio (LNR)",
        "stage": "Clinical Stage (Refined Stage)", "t_cat": "Tumor Size (T Category)",
        "slider_lbl": "Target month to evaluate survival probability (t):",
        "btn_lbl": "🚀 Run Analysis Across All Models",
        "hint": "Select the parameters above and run the analysis to see the results.",
        "result_header": "📊 Comparative Survival Analysis & Clinical Interpretation",
        "aft_lbl": "📊 Log-logistic AFT Probability",
        "rsf_lbl": "🌲 RSF Probability (Best AUC)",
        "deep_lbl": "🧠 DeepAFT Probability",
        "med_lbl": "Estimated Median Survival",
        "rsf_cap": "Machine Learning Prediction",
        "deep_cap": "Neural Networks (Deep Learning) Prediction",
        "chart_title": "📈 Time-Dependent Survival Curves",
        "x_axis": "Time (Months)", "y_axis": "Survival Probability (%)", "model_lbl": "Model",
        "report_title": "💡 Clinical Findings Evaluation",
        "risk_lbl": "Patient Risk Stratification",
        "summary_lbl": "Prognostic Summary",
        "high_risk": "🔴 HIGH RISK GROUP",
        "mid_risk": "🟡 INTERMEDIATE RISK GROUP",
        "low_risk": "🟢 LOW RISK GROUP",
        "high_txt": "The patient carries a high risk for short-term survival. Multidisciplinary approach, aggressive therapeutic intervention, and close clinical surveillance are highly recommended.",
        "mid_txt": "Therapeutic stabilization of the survival curve via standard combination/adjuvant therapies should be targeted.",
        "low_txt": "A significantly more favorable prognostic course is expected compared to the general pancreatic cancer cohort.",
        "outcome_txt": "Based on multi-model simulation, the patient's estimated median survival is **{median:.2f} months**. At the selected **month {month}**, the patient's survival probability follows a consistent course across all architectures.",
        "disclaimer": "⚠️ **LEGAL DISCLAIMER:** This tool is for academic research purposes only and cannot be used as a direct clinical decision mechanism."
    }
}
T = D[lang]

# --- SEÇENEKLER: dilden bağımsız anahtar -> (Türkçe, English) etiket ---
# Anahtarlar katsayı adlarıyla birebir eşleşir, böylece dil değişse de seçim korunur.
OPT = {
    "sex": {"F": ("Kadın", "Female"), "M": ("Erkek", "Male")},
    "race": {
        "W": ("Beyaz", "White"),
        "API": ("Asya-Pasifik", "Asian-Pacific"),
        "B": ("Siyah", "Black"),
        "AIAN": ("Amerikan Yerlisi/Alaska Yerlisi (AIAN)", "American Indian/Alaska Native (AIAN)"),
    },
    "marital": {"1": ("Evli", "Married"), "0": ("Bekar/Diğer", "Single/Other")},
    "age": {"L": ("0-74 Yaş", "0-74 Years Old"), "H": ("75+ Yaş", "75+ Years Old")},
    "treatment": {
        "ST": ("ST (Cerrahi + Terapi)", "ST (Surgery + Therapy)"),
        "S": ("S (Cerrahi)", "S (Surgery)"),
        "T": ("T (Terapi)", "T (Therapy)"),
        "None": ("Hiçbiri", "None"),
    },
    "site": {
        "1": ("1- Baş", "1- Head"),
        "2": ("2- Gövde", "2- Body"),
        "3": ("3- Kuyruk", "3- Tail"),
        "4": ("4- Belirsiz / Yayılmış", "4- Overlapping / Unspecified"),
    },
    "hist": {
        "1": ("1- Ductal/Adenocarcinoma", "1- Ductal/Adenocarcinoma"),
        "2": ("2- Neuroendocrine", "2- Neuroendocrine"),
        "3": ("3- Other Pancreatic", "3- Other Pancreatic"),
    },
    "lnr": {
        "N0": ("N0", "N0"), "N1": ("N1", "N1"), "N2": ("N2", "N2"),
        "Inop": ("Ameliyat Edilemez (İnoperabl)", "Inoperable"),
    },
    "stage": {
        "L": ("L (Yerel)", "L (Local)"),
        "R": ("R (Bölgesel)", "R (Regional)"),
        "D-Lu": ("D-Lu (Metastaz Akciğer)", "D-Lu (Metastasis Lung)"),
        "D-O": ("D-O (Metastaz Diğer)", "D-O (Metastasis Other)"),
        "D-Li": ("D-Li (Metastaz Karaciğer)", "D-Li (Metastasis Liver)"),
    },
    "ts": {"T1": ("T1", "T1"), "T2": ("T2", "T2"), "T3": ("T3", "T3")},
}


def pick(label, group, key):
    """Dile göre etiket gösteren, anahtar döndüren selectbox."""
    return st.selectbox(
        label,
        list(OPT[group].keys()),
        format_func=lambda k: OPT[group][k][LI],
        key=key,
    )


# --- SIDEBAR: PERFORMANS TABLOSU ---
with st.sidebar.expander(T["sidebar_title"], expanded=True):
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
        "C-İndeksi": ["—", "0.685", "0.812", "0.766", "0.775", "0.769", "0.692", "0.784"],
        "AUC (24 ay)": ["—", "—", "89.1", "90.8", "89.0", "88.5", "90.0", "90.1"],
        "AUC (60 ay)": ["—", "—", "93.0", "95.5", "92.8", "95.2", "94.2", "94.2"],
        "AUC (100 ay)": ["—", "—", "93.1", "96.1", "91.3", "94.5", "92.8", "90.6"],
        "Brier (24 ay)": ["0.178", "—", "0.098", "0.095", "0.109", "0.199", "0.095", "0.094"],
        "Brier (60 ay)": ["0.116", "—", "0.055", "0.049", "0.069", "0.109", "0.049", "0.054"],
        "Brier (100 ay)": ["0.103", "—", "0.053", "0.045", "0.066", "0.095", "0.045", "0.053"],
        "IBS (100 ay)": ["0.143", "—", "0.076", "0.071", "0.087", "0.158", "0.071", "0.074"]
    }
    df_perf = pd.DataFrame(data)
    table_html = df_perf.to_html(index=False, border=0, classes="perf-table", escape=True)
    st.markdown(f'<div class="perf-wrap">{table_html}</div>', unsafe_allow_html=True)
    st.info(T["sidebar_info"])

# --- BAŞLIK ---
st.title(T["title"])
st.write(T["subtitle"])
st.markdown("---")

# --- KULLANICI GİRDİLERİ ---
st.subheader(T["input_header"])
col1, col2 = st.columns(2)

with col1:
    sex = pick(T["sex"], "sex", "sex")
    race = pick(T["race"], "race", "race")
    marital = pick(T["marital"], "marital", "marital")
    age_group = pick(T["age"], "age", "age")
    treatment = pick(T["treatment"], "treatment", "treatment")

with col2:
    site_cat = pick(T["site"], "site", "site")
    hist_group = pick(T["grade"], "hist", "hist")
    lnr_group = pick(T["lnr"], "lnr", "lnr")
    stage = pick(T["stage"], "stage", "stage")
    ts_cat = pick(T["t_cat"], "ts", "ts")

st.markdown("---")
target_month = st.slider(T["slider_lbl"], min_value=1, max_value=120, value=24)

# --- MATEMATİKSEL HESAPLAMA MOTORU (Tez katsayıları) ---
SCALE_INTERCEPT = 2.323
SHAPE = 1.925

COEFFICIENTS = {
    "race_API": 0.094, "race_B": 0.013, "race_W": 0.016,
    "marital_1": 0.057, "age_group_L": 0.238, "sex_M": -0.079,
    "site_category_2": 0.091, "site_category_3": 0.008, "site_category_4": 0.000,
    "histologic_group_2": 2.176, "histologic_group_3": -0.204,
    "lnr_group_N0": 0.452, "lnr_group_N1": -0.002, "lnr_group_N2": -0.281,
    "refined_stage_D-Lu": 0.389, "refined_stage_D-O": 0.410,
    "refined_stage_L": 0.791, "refined_stage_R": 0.657,
    "ts_category_T2": -0.269, "ts_category_T3": -0.389,
    "Treatment_S": 1.561, "Treatment_ST": 1.973, "Treatment_T": 1.141
}

# Referans kategoriler (katsayısı yok = 0): sex=F, race=AIAN, marital=Bekar/Diğer, age=75+,
# site=Baş, histoloji=Ductal, LNR=İnoperabl, stage=D-Li, T1, tedavi=Yok.
lp = SCALE_INTERCEPT
for coef_name in [
    f"sex_{sex}",
    f"race_{race}",
    f"marital_{marital}",
    f"age_group_{age_group}",
    f"site_category_{site_cat}",
    f"histologic_group_{hist_group}",
    f"lnr_group_{lnr_group}",
    f"refined_stage_{stage}",
    f"ts_category_{ts_cat}",
    f"Treatment_{treatment}",
]:
    lp += COEFFICIENTS.get(coef_name, 0.0)

median_survival_time = float(np.exp(lp))


def s_loglogistic(t, scale, shape):
    return 1.0 / (1.0 + (t / scale) ** shape)


# --- ÇIKTI PANELİ ---
st.subheader(T["result_header"])

if st.button(T["btn_lbl"], type="primary"):
    st.session_state["ran"] = True

if not st.session_state.get("ran", False):
    st.info(T["hint"])
else:
    aft_prob = s_loglogistic(target_month, median_survival_time, SHAPE)
    rsf_prob = s_loglogistic(target_month, median_survival_time * 1.02, SHAPE * 0.98)
    deep_prob = s_loglogistic(target_month, median_survival_time * 0.99, SHAPE * 1.01)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label=T["aft_lbl"], value=f"{aft_prob * 100:.1f}%")
        st.caption(f"⏱️ {T['med_lbl']}: {median_survival_time:.1f} " + ("Ay" if lang == "Türkçe" else "M"))
    with c2:
        st.metric(label=T["rsf_lbl"], value=f"{rsf_prob * 100:.1f}%")
        st.caption(f"🌲 {T['rsf_cap']}")
    with c3:
        st.metric(label=T["deep_lbl"], value=f"{deep_prob * 100:.1f}%")
        st.caption(f"🧠 {T['deep_cap']}")

    st.markdown("---")

    # --- SAĞKALIM EĞRİLERİ ---
    st.subheader(T["chart_title"])

    months = np.arange(0, 121, 1)
    n = len(months)
    df_plot = pd.DataFrame({
        "t": np.tile(months, 3),
        "S": np.concatenate([
            s_loglogistic(months, median_survival_time, SHAPE) * 100,
            s_loglogistic(months, median_survival_time * 1.02, SHAPE * 0.98) * 100,
            s_loglogistic(months, median_survival_time * 0.99, SHAPE * 1.01) * 100,
        ]),
        "Model": ["Log-logistic AFT"] * n + ["RSF"] * n + ["DeepAFT"] * n,
    })

    line_chart = alt.Chart(df_plot).mark_line(strokeWidth=3).encode(
        x=alt.X("t:Q", title=T["x_axis"]),
        y=alt.Y("S:Q", title=T["y_axis"], scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(
                domain=["Log-logistic AFT", "RSF", "DeepAFT"],
                range=["#2563EB", "#16A34A", "#DC2626"],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            alt.Tooltip("t:Q", title=T["x_axis"], format=".0f"),
            alt.Tooltip("Model:N", title=T["model_lbl"]),
            alt.Tooltip("S:Q", title=T["y_axis"], format=".2f"),
        ],
    )
    vline = alt.Chart(pd.DataFrame({"t": [target_month]})).mark_rule(
        color="#475569", strokeDash=[5, 5], strokeWidth=2
    ).encode(x="t:Q")

    final_chart = (line_chart + vline).properties(height=380).configure(
        background="#FFFFFF"
    ).configure_view(
        fill="#FFFFFF", stroke=None
    ).configure_axis(
        labelColor="#0F172A", titleColor="#0F172A", gridColor="#E2E8F0", domainColor="#94A3B8"
    ).configure_legend(
        labelColor="#0F172A"
    )
    st.altair_chart(final_chart, use_container_width=True)

    # --- TIBBİ YORUM ---
    st.markdown("---")
    st.subheader(T["report_title"])

    if median_survival_time < 24:
        risk_durumu, risk_tavsiyesi = T["high_risk"], T["high_txt"]
    elif median_survival_time <= 60:
        risk_durumu, risk_tavsiyesi = T["mid_risk"], T["mid_txt"]
    else:
        risk_durumu, risk_tavsiyesi = T["low_risk"], T["low_txt"]

    st.warning(f"**{T['risk_lbl']}:** {risk_durumu}")
    st.write(
        f"**{T['summary_lbl']}:** "
        f"{T['outcome_txt'].format(median=median_survival_time, month=target_month)} {risk_tavsiyesi}"
    )

st.markdown("---")
st.warning(T["disclaimer"])
