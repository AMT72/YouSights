import os
import re
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gradio as gr
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# في بداية الملف مع بقية التعريفات
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# هنا نقرأ الـ ID من ملف البيئة، وإذا لم يوجد نستخدم القيمة الافتراضية الخاصة بك
TEMPLATE_ID = os.getenv("GOOGLE_SLIDES_TEMPLATE_ID", "1c8OBhWTGmKvjFsMZ7NYuz0a7kadOmf6505S967UVufQ")

# ✅ 2. إعداد نموذج تحليل المشاعر (CAMeL-Lab)
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HF_TOKEN)
device = 0 if torch.cuda.is_available() else -1
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

# ✅ 3. دوال المساعدة للتعامل مع يوتيوب والبيانات
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    return match.group(1) if match else url

def get_comments(video_id, max_comments=100):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
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
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "User": snippet["authorDisplayName"],
                "Comment": snippet["textDisplay"],
                "Likes": snippet["likeCount"]
            })
        next_page_token = response.get("nextPageToken")
        if not next_page_token: break
    return pd.DataFrame(comments)

# ✅ 4. الدالة الأساسية للتحليل (Logic)
def analyze_youtube_video(video_url):
    video_id = extract_video_id(video_url)
    df = get_comments(video_id)
    
    # تحليل المشاعر باستخدام الموديل
    df['Sentiment'] = df['Comment'].apply(lambda x: classifier(x[:512])[0]['label'] if x else 'neutral')
    
    # رسم المخطط البياني وتنسيقه
    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
    labels = ['positive', 'neutral', 'negative']
    sizes = [sentiment_counts.get(l, 0) for l in labels]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']

    fig, ax = plt.subplots(facecolor='black')
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.axis('equal')
    for text in ax.texts:
        text.set_color('white')
    
    output_img = "sentiment_pie.png"
    plt.savefig(output_img, bbox_inches='tight', facecolor='black')
    
    # ملاحظة: كود الـ Google Slides يحتاج لمصادقة OAuth كاملة للعمل خارج Colab
    # حالياً سيعيد الكود ملف الرسم البياني كإثبات عمل (Demo)
    return output_img

# ✅ 5. واجهة المستخدم (Gradio)
iface = gr.Interface(
    fn=analyze_youtube_video,
    inputs=gr.Textbox(label="أدخل رابط فيديو يوتيوب"),
    outputs=gr.Image(label="نتائج تحليل المشاعر"),
    title="YouTube Arabic Sentiment Analyzer",
    description="قم بتحليل تعليقات اليوتيوب العربية باستخدام نموذج BERT واستخرج تقارير مرئية."
)

if __name__ == "__main__":
    iface.launch()
