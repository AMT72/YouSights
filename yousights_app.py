import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import os
import io
import time
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="YouSights",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — dark teal theme matching PDF ──────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --teal-dark:  #0a2e2e;
    --teal-mid:   #0d4a4a;
    --teal-light: #0e6b6b;
    --cyan:       #00d4c8;
    --cyan-soft:  #4dd9d0;
    --sky:        #7ed8f6;
    --white:      #f0fafa;
    --muted:      #7ba8a8;
    --card-bg:    rgba(13,74,74,0.45);
    --border:     rgba(0,212,200,0.18);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--teal-dark);
    color: var(--white);
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061c1c 0%, #0a2e2e 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--white) !important; }

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(0,212,200,0.08) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--white) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > label, .stNumberInput > label,
.stSelectbox > label { color: var(--muted) !important; font-size: 0.82rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4c8, #0e6b6b) !important;
    color: #061c1c !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    font-size: 1rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,212,200,0.35) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(8px);
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--cyan) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* Progress */
.stProgress > div > div { background: var(--cyan) !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #061c1c 0%, #0d4a4a 60%, #0e6b6b 100%);
    border: 1px solid rgba(0,212,200,0.2);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
">
    <div style="
        position:absolute; top:-60px; right:-60px;
        width:220px; height:220px;
        border-radius:50%;
        background: radial-gradient(circle, rgba(0,212,200,0.12) 0%, transparent 70%);
    "></div>
    <div style="display:flex; align-items:center; gap:1.2rem;">
        <div style="
            background: linear-gradient(135deg, #00d4c8, #0e6b6b);
            border-radius: 16px;
            width: 56px; height: 56px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.8rem;
        ">🎯</div>
        <div>
            <h1 style="
                font-family:'Syne',sans-serif;
                font-weight:800; font-size:2rem;
                margin:0; color:#f0fafa;
                letter-spacing:-0.5px;
            ">YouSights</h1>
            <p style="margin:0; color:#7ba8a8; font-size:0.88rem;">
                Discover. Analyze. Understand your audience.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800; color:#00d4c8;">
            🎯 YouSights
        </div>
        <div style="font-size:0.75rem; color:#7ba8a8; margin-top:0.3rem;">v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")

    yt_api_key = st.text_input(
        "YouTube API Key",
        type="password",
        placeholder="AIza...",
        help="Get yours at console.cloud.google.com"
    )

    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        placeholder="hf_...",
        help="Required to load Arabic sentiment model"
    )

    st.markdown("---")
    st.markdown("### 📹 Video")

    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=..."
    )

    max_comments = st.number_input(
        "Max Comments to Fetch",
        min_value=20,
        max_value=500,
        value=100,
        step=20
    )

    top_n = st.slider("Top Comments to Show", 3, 10, 5)

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Video")

    st.markdown("""
    <div style="
        margin-top:2rem;
        padding:1rem;
        background:rgba(0,212,200,0.06);
        border:1px solid rgba(0,212,200,0.15);
        border-radius:10px;
        font-size:0.78rem;
        color:#7ba8a8;
        line-height:1.6;
    ">
        <b style="color:#00d4c8">How it works</b><br>
        1. Enter your API keys<br>
        2. Paste a YouTube URL<br>
        3. Hit Analyze<br>
        4. Download your report
    </div>
    """, unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    return match.group(1) if match else url


@st.cache_resource(show_spinner=False)
def load_sentiment_model(token):
    from huggingface_hub import login
    login(token=token)
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name)
    device     = 0 if torch.cuda.is_available() else -1
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)


def get_comments(video_id, api_key, max_comments=100):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments, next_page_token = [], None
    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order="relevance"
        ).execute()
        for item in response["items"]:
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "User":    s["authorDisplayName"],
                "Comment": s["textDisplay"],
                "Likes":   s["likeCount"]
            })
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return pd.DataFrame(comments)


def run_sentiment(df, classifier):
    results = []
    for text in df["Comment"]:
        try:
            label = classifier(str(text)[:512])[0]["label"]
        except Exception:
            label = "neutral"
        results.append(label)
    df["Sentiment"] = results
    return df


def sentiment_emoji(label):
    return {"positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "🟡 Neutral"}.get(label, label)


def make_pie_chart(counts):
    labels = ["positive", "neutral", "negative"]
    values = [counts.get(l, 0) for l in labels]
    colors_list = ["#00d4c8", "#7ed8f6", "#e74c3c"]
    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels],
        values=values,
        hole=0.52,
        marker=dict(colors=colors_list, line=dict(color="#0a2e2e", width=3)),
        textfont=dict(family="DM Sans", size=13, color="#f0fafa"),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(family="DM Sans", color="#f0fafa", size=12),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(t=20, b=20, l=20, r=20),
        height=320
    )
    return fig


def make_bar_chart(counts):
    labels = ["Positive", "Neutral", "Negative"]
    values = [counts.get(l.lower(), 0) for l in labels]
    colors_list = ["#00d4c8", "#7ed8f6", "#e74c3c"]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors_list,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="Syne", color="#f0fafa", size=13)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#7ba8a8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,212,200,0.1)", color="#7ba8a8", range=[0, max(values)+15]),
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        showlegend=False
    )
    return fig


def generate_pdf(df, top_comments, sentiment_counts, video_id):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )

    TEAL   = colors.HexColor("#0a2e2e")
    CYAN   = colors.HexColor("#00d4c8")
    WHITE  = colors.HexColor("#f0fafa")
    MUTED  = colors.HexColor("#7ba8a8")
    RED    = colors.HexColor("#e74c3c")
    SKY    = colors.HexColor("#7ed8f6")
    DARK   = colors.HexColor("#061c1c")

    def _p(name, **kw):
        return ParagraphStyle(name, **kw)

    title_s   = _p("T",  fontSize=28, textColor=CYAN,  fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4)
    sub_s     = _p("S",  fontSize=11, textColor=MUTED, fontName="Helvetica",      alignment=TA_CENTER, spaceAfter=16)
    h1_s      = _p("H1", fontSize=16, textColor=WHITE, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=8)
    body_s    = _p("B",  fontSize=10, textColor=WHITE, fontName="Helvetica",      leading=15, spaceAfter=4)
    small_s   = _p("SM", fontSize=8,  textColor=MUTED, fontName="Helvetica",      spaceAfter=2)
    num_s     = _p("N",  fontSize=22, textColor=CYAN,  fontName="Helvetica-Bold", alignment=TA_CENTER)
    rank_s    = _p("R",  fontSize=13, textColor=DARK,  fontName="Helvetica-Bold", alignment=TA_CENTER)

    story = []

    # ── Cover block ──
    cover = Table([[
        Paragraph("🎯 YouSights", title_s),
    ]], colWidths=[25*cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), TEAL),
        ("TOPPADDING",   (0,0), (-1,-1), 28),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [12]),
    ]))
    story.append(cover)
    story.append(Paragraph("Discover. Analyze. Understand your audience.", sub_s))
    story.append(Paragraph(f"YouTube Comments Report  •  Video ID: {video_id}", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=CYAN, spaceAfter=18))

    # ── Sentiment overview ──
    story.append(Paragraph("Sentiment Overview", h1_s))

    pos = sentiment_counts.get("positive", 0)
    neu = sentiment_counts.get("neutral",  0)
    neg = sentiment_counts.get("negative", 0)

    sent_data = [[
        Paragraph(f"{pos:.1f}%", num_s),
        Paragraph(f"{neu:.1f}%", num_s),
        Paragraph(f"{neg:.1f}%", num_s),
    ],[
        Paragraph("Positive", small_s),
        Paragraph("Neutral",  small_s),
        Paragraph("Negative", small_s),
    ]]
    sent_tbl = Table(sent_data, colWidths=[8*cm, 8*cm, 8*cm])
    sent_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), colors.HexColor("#0d4a4a")),
        ("BACKGROUND",   (1,0), (1,-1), colors.HexColor("#0d4a4a")),
        ("BACKGROUND",   (2,0), (2,-1), colors.HexColor("#0d4a4a")),
        ("TOPPADDING",   (0,0), (-1,-1), 12),
        ("BOTTOMPADDING",(0,0), (-1,-1), 12),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#0e6b6b")),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(sent_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Top comments ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#0e6b6b"), spaceAfter=12))
    story.append(Paragraph(f"Top {len(top_comments)} Most Liked Comments", h1_s))

    for i, row in enumerate(top_comments.itertuples(), 1):
        sentiment = getattr(row, "Sentiment", "neutral")
        s_color   = CYAN if sentiment == "positive" else (RED if sentiment == "negative" else SKY)
        row_data  = [[
            Paragraph(str(i), rank_s),
            Paragraph(str(row.Comment)[:300], body_s),
            Paragraph(f"👍 {row.Likes:,}", _p("L", fontSize=13, textColor=CYAN, fontName="Helvetica-Bold", alignment=TA_RIGHT)),
            Paragraph(sentiment.upper(), _p("SN", fontSize=9, textColor=s_color, fontName="Helvetica-Bold", alignment=TA_CENTER)),
        ]]
        row_tbl = Table(row_data, colWidths=[1.2*cm, 17*cm, 3.5*cm, 3*cm])
        row_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,0), CYAN),
            ("BACKGROUND",    (1,0), (-1,0), colors.HexColor("#0d4a4a")),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ROUNDEDCORNERS", [6]),
        ]))
        story.append(row_tbl)
        story.append(Spacer(1, 0.25*cm))

    # ── Footer ──
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#0e6b6b")))
    story.append(Paragraph(
        "Generated by YouSights v2.0  •  Developed by Azzam",
        _p("F", fontSize=8, textColor=MUTED, fontName="Helvetica", alignment=TA_CENTER, spaceBefore=6)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ── Main analysis ───────────────────────────────────────────────
if analyze_btn:
    if not yt_api_key or not hf_token or not video_url:
        st.error("⚠️ Please fill in API Key, HuggingFace Token, and YouTube URL.")
        st.stop()

    video_id = extract_video_id(video_url)

    # Step 1 — Load model
    with st.status("🤖 Loading Arabic sentiment model...", expanded=True) as status:
        classifier = load_sentiment_model(hf_token)
        status.update(label="✅ Model ready", state="complete")

    # Step 2 — Fetch comments
    with st.status("📥 Fetching comments from YouTube...", expanded=True) as status:
        try:
            df = get_comments(video_id, yt_api_key, max_comments)
            st.write(f"✅ Fetched {len(df)} comments")
            status.update(label=f"✅ Got {len(df)} comments", state="complete")
        except Exception as e:
            st.error(f"❌ YouTube API error: {e}")
            st.stop()

    # Step 3 — Analyze sentiment
    with st.status("🧠 Analyzing sentiment...", expanded=True) as status:
        prog = st.progress(0)
        results = []
        for i, text in enumerate(df["Comment"]):
            try:
                label = classifier(str(text)[:512])[0]["label"]
            except Exception:
                label = "neutral"
            results.append(label)
            if i % 10 == 0:
                prog.progress((i + 1) / len(df))
        df["Sentiment"] = results
        prog.progress(1.0)
        status.update(label="✅ Sentiment analysis done", state="complete")

    st.session_state["df"]       = df
    st.session_state["video_id"] = video_id
    st.session_state["top_n"]    = top_n
    st.rerun()


# ── Results display ─────────────────────────────────────────────
if "df" in st.session_state:
    df       = st.session_state["df"]
    video_id = st.session_state["video_id"]
    top_n    = st.session_state.get("top_n", 5)

    counts_raw = df["Sentiment"].value_counts(normalize=True) * 100
    counts     = counts_raw.to_dict()
    top_comments = df.sort_values("Likes", ascending=False).head(top_n)
    total      = len(df)

    # ── Metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("💬 Total Comments",  f"{total:,}")
    with m2: st.metric("🟢 Positive",        f"{counts.get('positive', 0):.1f}%")
    with m3: st.metric("🔴 Negative",        f"{counts.get('negative', 0):.1f}%")
    with m4: st.metric("👍 Max Likes",       f"{df['Likes'].max():,}")

    st.markdown("<div style='margin: 1.5rem 0 0.5rem; height:1px; background:rgba(0,212,200,0.15);'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Analysis", "💬 Top Comments", "📋 All Comments"])

    # ── Tab 1: Charts ──
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="font-family:'DM Sans';font-size:0.8rem;color:#7ba8a8;margin-bottom:0.5rem;">
                SENTIMENT DISTRIBUTION
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(make_pie_chart(counts), use_container_width=True)

        with col2:
            st.markdown("""
            <div style="font-family:'DM Sans';font-size:0.8rem;color:#7ba8a8;margin-bottom:0.5rem;">
                PERCENTAGE BREAKDOWN
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(make_bar_chart(counts), use_container_width=True)

        # Sentiment over comments (trend)
        st.markdown("""
        <div style="font-family:'DM Sans';font-size:0.8rem;color:#7ba8a8;margin:1rem 0 0.5rem;">
            SENTIMENT TREND (rolling average)
        </div>""", unsafe_allow_html=True)

        sent_map  = {"positive": 1, "neutral": 0, "negative": -1}
        df["sent_num"] = df["Sentiment"].map(sent_map)
        df["rolling"]  = df["sent_num"].rolling(10, min_periods=1).mean()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            y=df["rolling"],
            mode="lines",
            line=dict(color="#00d4c8", width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(0,212,200,0.08)",
            name="Sentiment trend"
        ))
        fig_trend.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#7ba8a8", title="Comment #"),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,212,200,0.08)", color="#7ba8a8",
                       tickvals=[-1, 0, 1], ticktext=["Negative", "Neutral", "Positive"]),
            height=260,
            margin=dict(t=10, b=30, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Tab 2: Top comments ──
    with tab2:
        st.markdown(f"""
        <div style="
            font-family:'Syne',sans-serif;
            font-size:1.15rem; font-weight:700;
            color:#f0fafa; margin-bottom:1.2rem;
        ">🏆 Top {top_n} Most Liked Comments</div>
        """, unsafe_allow_html=True)

        for i, row in enumerate(top_comments.itertuples(), 1):
            sent   = getattr(row, "Sentiment", "neutral")
            s_color = {"positive":"#00d4c8","negative":"#e74c3c","neutral":"#7ed8f6"}.get(sent,"#7ba8a8")
            s_icon  = {"positive":"🟢","negative":"🔴","neutral":"🟡"}.get(sent,"⚪")

            st.markdown(f"""
            <div style="
                background: rgba(13,74,74,0.4);
                border: 1px solid rgba(0,212,200,0.15);
                border-left: 3px solid {s_color};
                border-radius: 12px;
                padding: 1rem 1.3rem;
                margin-bottom: 0.8rem;
                display: flex;
                gap: 1rem;
                align-items: flex-start;
            ">
                <div style="
                    background: linear-gradient(135deg,#00d4c8,#0e6b6b);
                    border-radius: 50%;
                    min-width: 36px; height: 36px;
                    display:flex; align-items:center; justify-content:center;
                    font-family:'Syne',sans-serif;
                    font-weight:800; color:#061c1c; font-size:0.95rem;
                ">{i}</div>
                <div style="flex:1;">
                    <div style="font-size:0.95rem; color:#f0fafa; line-height:1.6; margin-bottom:0.4rem;">
                        {row.Comment}
                    </div>
                    <div style="display:flex; gap:1.2rem; align-items:center;">
                        <span style="font-family:'Syne',sans-serif; font-weight:700; color:#00d4c8; font-size:0.9rem;">
                            👍 {row.Likes:,} likes
                        </span>
                        <span style="font-size:0.78rem; color:#7ba8a8;">
                            {s_icon} {sent.capitalize()}
                        </span>
                        <span style="font-size:0.78rem; color:#7ba8a8;">
                            by {row.User}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 3: All comments table ──
    with tab3:
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            search = st.text_input("🔍 Search comments", placeholder="Type to filter...")
        with col_f2:
            sent_filter = st.selectbox("Filter by sentiment", ["All", "Positive", "Neutral", "Negative"])

        display_df = df.copy()
        if search:
            display_df = display_df[display_df["Comment"].str.contains(search, case=False, na=False)]
        if sent_filter != "All":
            display_df = display_df[display_df["Sentiment"] == sent_filter.lower()]

        display_df["Sentiment"] = display_df["Sentiment"].map(sentiment_emoji)
        st.dataframe(
            display_df[["User", "Comment", "Likes", "Sentiment"]].reset_index(drop=True),
            use_container_width=True,
            height=420
        )
        st.caption(f"Showing {len(display_df):,} of {total:,} comments")

    # ── Download section ──
    st.markdown("<div style='margin: 1.5rem 0 0.5rem; height:1px; background:rgba(0,212,200,0.15);'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; color:#f0fafa; margin-bottom:1rem;">
        📥 Download Report
    </div>
    """, unsafe_allow_html=True)

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        with st.spinner("Generating PDF..."):
            pdf_bytes = generate_pdf(df, top_comments, counts, video_id)
        st.download_button(
            "⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"yousights_{video_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with dl2:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"comments_{video_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with dl3:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Comments")
            top_comments.to_excel(writer, index=False, sheet_name="Top Comments")
        st.download_button(
            "⬇️ Download Excel",
            data=excel_buf.getvalue(),
            file_name=f"yousights_{video_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # ── New analysis button ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Analyze Another Video"):
        for k in ["df", "video_id", "top_n"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Empty state ─────────────────────────────────────────────────
elif not analyze_btn:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 5rem 2rem;
        color: #7ba8a8;
    ">
        <div style="font-size:4rem; margin-bottom:1rem;">🎯</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#f0fafa; margin-bottom:0.6rem;">
            Ready to analyze
        </div>
        <div style="font-size:0.9rem; max-width:400px; margin:0 auto; line-height:1.7;">
            Enter your API keys and a YouTube URL in the sidebar, then hit <b style="color:#00d4c8">Analyze Video</b> to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)
