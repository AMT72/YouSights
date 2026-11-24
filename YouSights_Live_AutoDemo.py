"""YouSights — deployable Streamlit companion to YouSights_Final_Project.ipynb.
Predictions are produced by a multilingual sentiment model, never hand-entered.
"""
import io
import re
import html
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="YouSights | Audience Intelligence", page_icon="🎯", layout="wide")
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
PALETTE = {"positive":"#23c8ad", "neutral":"#f1c66f", "negative":"#ff7285"}
ORDER = ["positive", "neutral", "negative"]
DEMO = [
    "الصوت مو واضح في المقطع", "شرحك ممتاز واستفدت كثير", "المونتاج جدا مرتب", "الفيديو طويل وممل",
    "التصوير سيء للأسف", "متى ينزل الجزء الثاني؟", "الترجمة ممتازة وشكرا لكم", "ليش الصوت منخفض؟",
    "المحتوى رهيب بس الإضاءة ضعيفة", "تجربة جميلة وأفكار مفيدة", "ياخي الصوت يقطع كثير", "الفيديو مرررة مفيد",
    "Great explanation, thanks!", "The microphone is too quiet", "Love the editing and transitions",
    "This video was a waste of time", "The camera quality looks fantastic", "Subtitles are out of sync",
    "Very informative and easy to understand", "Sound quality needs improvement", "The tutorial is amazing",
    "الصوت clear بس المونتاج ضعيف", "Great video بس الصوت سيئ", "الشرح good والموسيقى مزعجة"
]
st.markdown("""
<style>
.stApp {background:#061d29;color:#e8fffa}
.block-container{padding-top:2rem;max-width:1300px}
[data-testid="stSidebar"]{background:#082935}
h1,h2,h3{color:#d9fff6}
.hero{background:linear-gradient(110deg,#062333,#09646b);border:1px solid #167a7a;border-radius:22px;padding:30px 35px;color:white;margin-bottom:24px}
.hero h1{font-size:43px;margin:0;color:#dbfff9}.hero small{color:#7df2df;letter-spacing:2px}
.note{background:#173946;padding:12px 16px;border-radius:12px;border-left:4px solid #f1c66f;margin:12px 0}
</style>
<div class="hero"><small>REAL TRANSFORMER INFERENCE</small><h1>🎯 YouSights</h1><p>Bilingual YouTube Audience Intelligence — Discover · Analyze · Understand</p></div>
""", unsafe_allow_html=True)


def video_id_from_url(value):
    value=str(value).strip()
    if re.fullmatch(r'[\w-]{11}',value):return value
    p=urlparse(value if '://' in value else 'https://'+value)
    host=(p.hostname or '').lower()
    if host in {'youtu.be','www.youtu.be'}:
        vid=p.path.strip('/').split('/')[0]
    elif host in {'youtube.com','www.youtube.com','m.youtube.com'}:
        if p.path.rstrip('/')=='/watch':vid=parse_qs(p.query).get('v',[''])[0]
        else:
            bits=p.path.strip('/').split('/')
            vid=bits[1] if len(bits)>1 and bits[0] in ('shorts','embed','live') else ''
    else:vid=''
    if not re.fullmatch(r'[\w-]{11}',vid):raise ValueError('Invalid YouTube video URL or ID.')
    return vid


def from_youtube(url,api_key,limit):
    youtube=build('youtube','v3',developerKey=api_key,cache_discovery=False)
    vid=video_id_from_url(url)
    rows=[];page=None
    while len(rows)<limit:
        kw=dict(part='snippet',videoId=vid,textFormat='plainText',order='time',maxResults=min(100,limit-len(rows)))
        if page:kw['pageToken']=page
        data=youtube.commentThreads().list(**kw).execute()
        for item in data.get('items',[]):
            s=item['snippet']['topLevelComment']['snippet']
            rows.append({'Comment':s.get('textDisplay',''),'Likes':s.get('likeCount',0),
                         'User':s.get('authorDisplayName','Anonymous')})
        page=data.get('nextPageToken')
        if not page or not data.get('items'):break
    if not rows:raise ValueError('No comments returned. The video may have comments disabled.')
    return pd.DataFrame(rows)


def normalize_df(raw):
    cols={str(c).strip().lower():c for c in raw.columns}
    t=next((cols[c] for c in ('comment','text','textdisplay','content','body') if c in cols),None)
    if t is None:raise ValueError('Please include a Comment or text column in CSV.')
    df=pd.DataFrame({'Comment':raw[t].fillna('').astype(str).str.strip()})
    for name,choices,default in [
        ('Likes',('likes','likecount'),0),('User',('user','author','authordisplayname'),'Unknown'),
        ('true_sentiment',('true_sentiment','gold_sentiment','actual_sentiment'),None)]:
        k=next((cols[c] for c in choices if c in cols),None)
        df[name]=raw[k] if k is not None else default
    df=df.loc[df['Comment'].str.len()>0].reset_index(drop=True)
    if len(df)==0:raise ValueError('No nonempty comments in the uploaded data.')
    df['Likes']=pd.to_numeric(df['Likes'],errors='coerce').fillna(0).clip(lower=0).astype(int)
    return df


def language_group(text):
    a=bool(re.search('[\u0600-\u06FF]',str(text)));e=bool(re.search('[A-Za-z]',str(text)))
    return 'Mixed' if a and e else 'Arabic-script' if a else 'Latin-script' if e else 'Other'


@st.cache_resource(show_spinner=False)
def load_model():
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    return tok,mdl


def classify(df):
    tok,mdl=load_model()
    predictions=[];scores=[];lengths=[]
    texts=df['Comment'].tolist()
    for i in range(0,len(texts),8):
        chunk=texts[i:i+8]
        clean=[' '.join('@user' if p.startswith('@') and len(p)>1 else 'http' if p.startswith('http') else p for p in x.split()) for x in chunk]
        enc=tok(clean,return_tensors='pt',padding=True,truncation=True,max_length=256)
        lengths.extend(enc['attention_mask'].sum(dim=1).tolist())
        with torch.inference_mode():prob=mdl(**enc).logits.softmax(dim=-1).cpu().numpy()
        for row in prob:
            ind=int(np.argmax(row));label=str(mdl.config.id2label[ind]).lower()
            if label not in ORDER:raise ValueError('Unexpected model label '+label)
            predictions.append(label);scores.append(float(row[ind]))
    out=df.copy();out['Sentiment']=predictions;out['Confidence']=scores
    out['TokenCount']=lengths;out['Language']=out['Comment'].map(language_group)
    return out


def style(fig,title):
    fig.update_layout(title=title, paper_bgcolor='#082934',plot_bgcolor='#082934',font_color='#e4fffa',
                      margin=dict(l=25,r=20,t=65,b=30),height=370)
    fig.update_xaxes(gridcolor='#194554');fig.update_yaxes(gridcolor='#194554')
    return fig


def build_charts(df):
    counts=df['Sentiment'].value_counts().reindex(ORDER,fill_value=0)
    a=px.pie(names=ORDER,values=[int(counts[i]) for i in ORDER],color=ORDER,color_discrete_map=PALETTE,hole=.59)
    a.update_traces(textinfo='percent+label')
    bdf=df.groupby(['Language','Sentiment']).size().reset_index(name='Comments')
    b=px.bar(bdf,x='Language',y='Comments',color='Sentiment',color_discrete_map=PALETTE,barmode='stack')
    c=px.box(df,x='Sentiment',y='Confidence',color='Sentiment',color_discrete_map=PALETTE,points='all')
    c.update_yaxes(tickformat='.0%',range=[0,1.05])
    d=px.scatter(df,x='TokenCount',y='Confidence',color='Sentiment',color_discrete_map=PALETTE,
                 size=df['Likes'].clip(lower=0)+1,size_max=24,hover_data=['Comment','Likes'])
    d.update_yaxes(tickformat='.0%',range=[0,1.05])
    return [style(a,'01 · Sentiment share'),style(b,'02 · By writing system'),
            style(c,'03 · Model confidence'),style(d,'04 · Token length × confidence')]


with st.sidebar:
    st.subheader('⚙️ Configure analysis')
    mode=st.radio('Data source',['Demo (synthetic)','Upload CSV','YouTube API'])
    csv_file=None;key='';url='';limit=100
    if mode=='Upload CSV':
        csv_file=st.file_uploader('Upload CSV (Comment or text column)',type='csv')
    if mode=='YouTube API':
        url=st.text_input('YouTube video URL')
        key=st.text_input('YouTube Data API key',type='password',help='Kept in current form only; not saved in source code.')
        limit=st.slider('Top-level comments to retrieve',20,300,100,step=20)
    st.info('The dashboard loads a demo automatically. For a real analysis, choose Upload CSV or YouTube API and click Analyze. Synthetic comments are NOT real audience data.')
    start=st.button('🚀 Analyze selected source',type='primary',use_container_width=True)

# Start the DEMO automatically on first visit: no API key, upload, or click needed.
# Predictions are computed by the real XLM-T model at runtime.
auto_demo = 'analyzed' not in st.session_state
if start or auto_demo:
    try:
        if auto_demo or mode=='Demo (synthetic)':
            raw=pd.DataFrame({'Comment':DEMO,'Likes':[(i*7+3)%27 for i in range(len(DEMO))]})
            source='DEMO — synthetic comments'
        elif mode=='Upload CSV':
            if csv_file is None:raise ValueError('Please upload CSV first.')
            raw=pd.read_csv(csv_file);source=f'CSV — {csv_file.name}'
        else:
            if not key or not url:raise ValueError('Enter YouTube URL and API key.')
            raw=from_youtube(url,key,limit);source='YouTube API — '+video_id_from_url(url)
        with st.spinner('Loading model and computing predictions...'):
            analyzed=classify(normalize_df(raw))
        st.session_state['analyzed']=analyzed
        st.session_state['source']=source
    except Exception as exc:
        st.error(f'Analysis error: {exc}')

if 'analyzed' in st.session_state:
    df=st.session_state['analyzed'];source=st.session_state['source']
    st.caption('Source: '+source+' | Checkpoint: '+MODEL_NAME)
    if source.startswith('DEMO'):
        st.warning('DEMONSTRATION: These comments and likes are synthetic. Sentiment predictions are made live by a pretrained multilingual Transformer, NOT typed in by us. This does not measure a real audience.')
    counts=df['Sentiment'].value_counts()
    k1,k2,k3,k4=st.columns(4)
    k1.metric('💬 Comments',str(len(df)))
    k2.metric('🟢 Positive',f'{counts.get("positive",0)/len(df):.1%}')
    k3.metric('🟡 Neutral',f'{counts.get("neutral",0)/len(df):.1%}')
    k4.metric('🔴 Negative',f'{counts.get("negative",0)/len(df):.1%}')
    figs=build_charts(df)
    x1,x2=st.columns(2)
    with x1:st.plotly_chart(figs[0],use_container_width=True)
    with x2:st.plotly_chart(figs[1],use_container_width=True)
    x3,x4=st.columns(2)
    with x3:st.plotly_chart(figs[2],use_container_width=True)
    with x4:st.plotly_chart(figs[3],use_container_width=True)
    tab1,tab2=st.tabs(['📋 All comments','🏆 Most liked & uncertain'])
    with tab1:
        st.dataframe(df[['Comment','Likes','Language','Sentiment','Confidence','TokenCount']],use_container_width=True,hide_index=True)
    with tab2:
        st.markdown('**Most liked**')
        st.dataframe(df.nlargest(7,'Likes')[['Comment','Likes','Sentiment']],hide_index=True,use_container_width=True)
        st.markdown('**Lowest confidence — inspect manually**')
        st.dataframe(df.nsmallest(7,'Confidence')[['Comment','Confidence','Sentiment']],hide_index=True,use_container_width=True)

    true=df['true_sentiment'].fillna('').astype(str).str.strip().str.lower()
    if true.isin(ORDER).any():
        from sklearn.metrics import accuracy_score,f1_score
        mask=true.isin(ORDER)
        st.markdown(f'**Human-labelled evaluation:** n={int(mask.sum())}; '
                    f'Accuracy={accuracy_score(true[mask],df.loc[mask,"Sentiment"]):.3f}; '
                    f'Macro-F1={f1_score(true[mask],df.loc[mask,"Sentiment"],average="macro",labels=ORDER,zero_division=0):.3f}')
    else:
        st.caption('Accuracy / Macro-F1: N/A — no human-labeled true_sentiment values supplied.')

    st.download_button('⬇️ Download predictions CSV',df.to_csv(index=False).encode('utf-8-sig'),
                       file_name='YouSights_predictions.csv',mime='text/csv')
    plot_html=''.join(fig.to_html(full_html=False,include_plotlyjs=(i==0)) for i,fig in enumerate(figs))
    snapshot=f'<!doctype html><html><head><meta charset="utf-8"><title>YouSights report</title></head><body style="background:#061d29;color:white;font-family:Arial"><h1>🎯 YouSights</h1><p>Source: {html.escape(source)} — model predictions, not ground truth.</p>{plot_html}</body></html>'
    st.download_button('⬇️ Download interactive HTML report',snapshot.encode('utf-8'),
                       file_name='YouSights_Interactive_Report.html',mime='text/html')
else:
    st.markdown('### Ready to analyze 💬')
    st.write('Select **YouTube API** or **Upload CSV** to examine real comments, or use Demo to test the interface. Predictions come from XLM-T, a multilingual Transformer.')
