# UNSW-NB15 Network Intrusion Analytics Dashboard

A **single-file Streamlit data analytics dashboard** built on the UNSW-NB15 network intrusion dataset. This project performs exploratory data analysis (EDA) and visual storytelling across three interactive pages — no prediction model, analysis only.

---

## 📦 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | UNSW-NB15 Network Intrusion Dataset |
| **Source** | [Kaggle — mrwellsdavid/unsw-nb15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) |
| **File used** | `UNSW_NB15_training-set.csv` |
| **Records** | 82,332 network flow records |
| **Features** | 45 (numeric + categorical) |
| **Attack categories** | 9 attack types + Normal |

---

## 🚀 Quick Start

### 1. Clone / download

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Download the dataset

Download `UNSW_NB15_training-set.csv` from [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) and place it in the **same directory** as `dashboard.py`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
.
├── dashboard.py                   # Main Streamlit app (single file)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── UNSW_NB15_training-set.csv     # Dataset (download from Kaggle)
└── UNSW_NB15_Project_Report.docx  # Project report with analysis details
```

---

## 🖥️ Dashboard Pages

### 📊 Page 1 — Executive Overview
- **KPI cards**: total records, % attack vs normal, average attack duration
- **Top 3 attack categories** with flow counts and share percentages
- **Donut chart**: Normal vs Attack traffic split
- **Horizontal bar chart**: all attack category frequencies
- **Grouped bar chart**: top protocols broken down by traffic type
- **Box plot**: flow duration distributions by category
- **Summary statistics table** with per-category averages

### 🔍 Page 2 — Attack Type Analysis
- **Category filter** (multi-select) for focused exploration
- **Treemap**: attack volume proportions
- **Sunburst chart**: attack category → protocol drill-down
- **Protocol & service targeting**: top 10 horizontally ranked
- **Duration analysis**: mean vs median per attack type
- **Byte analysis**: source vs destination bytes stacked
- **Scatter plot**: packet rate vs duration coloured by category
- **Heatmap**: connection state × attack category cross-tabulation
- **Detailed statistics table** with protocol, service, state, averages

### ⚡ Page 3 — Risk, Opportunity & Action
- **Composite risk score** bar chart (volume + rate combined)
- **Threat radar** chart for the top-5 attack types
- **Protocol & service attack ratios** (% of flows that are attacks)
- **Anomaly indicator KPIs**: high-rate flows, zero-duration probes, packet-loss flows
- **Cumulative attack flow trend** vs record ID
- **Expandable opportunity cards** (7 recommendations, colour-coded by priority)
- **Prioritised action plan table** with timelines and expected impact
- **Feature correlation heatmap** (Pearson) across key numeric features

---

## 🔧 Data Cleaning Steps

1. Strip whitespace from all string columns
2. Standardise `attack_cat` values (blank → "Normal", " Backdoors" → "Backdoor")
3. Coerce numeric columns to float (invalid → NaN)
4. Fill NaN numeric values with column medians
5. Remove exact duplicate rows
6. Cast `label` to integer
7. Derive computed columns: `is_attack`, `total_bytes`, `total_pkts`

---

## 📋 Requirements

| Package | Version |
|---------|---------|
| streamlit | ≥ 1.35.0 |
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |
| plotly | ≥ 5.18.0 |
| python-docx | ≥ 1.1.0 |

Python 3.9+ recommended.

---

## 📄 License

This project is for educational and research purposes. The UNSW-NB15 dataset is provided by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra.
