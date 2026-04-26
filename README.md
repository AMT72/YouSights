# 🎯 YouSights

> Discover. Analyze. Understand your audience.

A Streamlit web app that fetches YouTube comments, runs Arabic sentiment analysis, and exports a professional PDF report — all in one click.

---

## ✨ Features

- 📥 **Fetch comments** from any YouTube video (up to 500 comments)
- 🧠 **Arabic sentiment analysis** powered by CAMeL-Lab BERT model
- 📊 **Interactive charts** — pie chart, bar chart, and sentiment trend
- 🏆 **Top N most liked comments** with sentiment labels
- 🔍 **Search & filter** across all comments
- 📄 **Download PDF report** matching the YouSights dark teal design
- 📊 **Download CSV / Excel** for further analysis

---

## 🖥️ Preview

| Section | What you see |
|---------|-------------|
| Sentiment Overview | Pie + bar charts with Positive / Neutral / Negative % |
| Trend Chart | Rolling sentiment score across all comments |
| Top Comments | Ranked by likes with sentiment badge |
| All Comments | Searchable, filterable table |
| Report | PDF with YouSights branding |

---

## 🚀 Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YouSights.git
cd YouSights
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🔑 API Keys Required

You'll enter both keys directly in the app sidebar — nothing is stored.

### YouTube Data API v3

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable **YouTube Data API v3**
4. Go to **Credentials → Create API Key**
5. Copy the key and paste it in the sidebar

### HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a **Read** token
3. Copy and paste it in the sidebar

> The token is needed to download the Arabic sentiment model on first run (~500MB, cached after that).

---

## 🤖 Sentiment Model

**CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment**

- Trained on mixed Arabic text (MSA + dialects)
- Labels: `positive`, `neutral`, `negative`
- Hosted on HuggingFace — loaded once and cached

---

## 🗂️ Project Structure

```
YouSights/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## 📋 Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| YouTube data | Google YouTube Data API v3 |
| Sentiment model | CAMeL-Lab Arabic BERT (HuggingFace) |
| Charts | Plotly |
| PDF report | ReportLab |
| Excel export | openpyxl |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: matplotlib` | `pip install matplotlib` |
| `YouTubeAPI quota exceeded` | Wait 24h or create a new API key |
| Model download slow | Normal on first run (~500MB) — cached after |
| Comments show 0 | Video may have comments disabled |
| PDF Arabic text reversed | Known ReportLab limitation — use the CSV export instead |

---

## 📄 License

MIT License — free to use and modify.

---

**Developed by Azzam** — YouSights v2.0
