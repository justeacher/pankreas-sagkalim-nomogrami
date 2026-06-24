# 🩺 Pancreatic Cancer Survival Nomogram & Clinical Decision Support System (CDSS)

An interactive, web-based clinical decision support tool designed for predicting time-dependent survival probabilities in pancreatic cancer patients. This system translates advanced survival analysis methodologies—specifically **Log-logistic Accelerated Failure Time (AFT)**, **Random Survival Forests (RSF)**, and **DeepAFT (Deep Learning)** architectures—into intuitive bedside clinical practice.

🚀 **[Click Here to Access the Live Web Application](YAYINA_ALDIĞINIZ_STREAMLIT_LİNKİNİ_BURAYA_YAPIŞTIRIN)**

---

## 📊 Overview & Model Performance

This repository accompanies a master's thesis and upcoming publication targeting high-impact statistics/medical informatics journals. The performance metrics computed across the patient cohort demonstrate robust prognostic discrimination and calibration:

| Algorithm / Method | C-Index | 24-Month AUC | 60-Month AUC | 100-Month AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Log-logistic AFT** | **0.8120** | 0.8930 | 0.9310 | 0.9280 |
| **Random Survival Forests (RSF)** | 0.7540 | **0.9110** | **0.9550** | **0.9550** |
| **DeepAFT (Deep Learning)** | 0.7855 | 0.9012 | 0.9420 | 0.9261 |
| Cox (Time-Transform) | 0.6850 | - | - | - |
| DeepSurv | 0.7700 | 0.8916 | 0.9309 | 0.9232 |
| DeepHit | 0.7642 | 0.8868 | 0.9512 | 0.9419 |
| N-MTLR | 0.6846 | 0.9018 | 0.9460 | 0.9283 |

*Key Insight:* The **Log-logistic AFT** model provides the highest global discriminative power (C-Index: 0.8120), while the **Random Survival Forests (RSF)** architecture excels in long-term predictions (60 & 100-Month AUC: 0.9550).

---

## 📋 Integrated Clinical Features

The interface dynamically evaluates ten patient-specific, non-linear, and high-dimensional clinicopathological risk factors:
* **Demographics:** Age Group, Sex, Marital Status, Race.
* **Tumor Characteristics:** Primary Tumor Site Category, Histologic Grade, Tumor Size (T-Category).
* **Staging & Lymph Nodes:** Refined Stage, Lymph Node Ratio (LNR) Group.
* **Intervention:** Therapeutic Treatment Protocol.

---

## 🛠️ Reproducibility & Local Deployment

To run this decision support system locally, clone this repository and install the dependencies:

```bash
git clone [https://github.com/justeacher/pankreas-sagkalim-nomogrami.git](https://github.com/justeacher/pankreas-sagkalim-nomogrami.git)
cd pankreas-sagkalim-nomogrami
pip install -r requirements.txt
streamlit run app.py
