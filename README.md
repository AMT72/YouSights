# 📊 YouSights: YouTube Comments Sentiment Analyzer

**YouSight** is a powerful tool designed to extract, analyze, and visualize the sentiment of YouTube comments. By leveraging the **Camel-BERT** model, it provides high accuracy in understanding various Arabic dialects, helping creators and researchers gain deeper insights into audience feedback.

---

## 🚀 Key Features

### 🔹 Automated Extraction

Fetches comments directly from YouTube using a video URL via the **YouTube Data API**.

### 🔹 Data Organization

Automatically saves extracted comments into structured **Excel files** for easy analysis.

### 🔹 Advanced AI Analysis

Utilizes **Camel-BERT** for precise sentiment classification:

* Positive
* Negative
* Neutral

Especially effective for Arabic dialects.

### 🔹 Insightful Reports

Generates ready-to-use reports including:

* Sentiment distribution percentages
* Analysis of most liked comments
* Visual representation of audience engagement

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** Streamlit (Web Interface)
* **AI Model:** Camel-BERT (Fine-tuned for Arabic Sentiment Analysis)
* **Data Handling:** Pandas, Openpyxl
* **Visualization:** Plotly / Matplotlib
* **APIs:** YouTube Data API v3

---

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/YouSight.git
cd YouSight
```

### 2. Install Dependencies

```bash
pip install -q streamlit pandas transformers torch openpyxl google-api-python-client
```

### 3. Set Up API Key

* Get your API Key from Google Cloud Console
* Add it to a configuration file or as an environment variable

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🖥️ How it Works

1. **Input:** Paste the YouTube video link into the dashboard
2. **Process:**

   * Fetch comments
   * Clean text
   * Run sentiment analysis using Camel-BERT
3. **Output:**

   * Generate a report
   * Display interactive charts showing audience "Vibe"

---

## 💡 Why Camel-BERT?

Unlike general-purpose models, **Camel-BERT** is trained on large-scale Arabic datasets, including both classical Arabic and regional dialects. This makes **YouSight** highly effective at understanding nuanced audience feedback across different Arabic-speaking regions (Saudi, Egyptian, Levantine, etc.).

---
