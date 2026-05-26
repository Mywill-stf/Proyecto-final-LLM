import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

# ── NLTK ──────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)

download_nltk()

# ── API Key ────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        load_dotenv()
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            st.error("No se encontró GROQ_API_KEY. Agrégala en .env o en Streamlit Secrets.")
            st.stop()
        return key

        
@st.cache_resource
def get_client():
    return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# ── NLP ────────────────────────────────────────────────────────────
STOP_ES = set(stopwords.words("spanish")) | {
    "más","si","así","aquí","ahí","ser","estar","hacer","tener","ir",
    "ver","muy","bien","mal","ya","también","sin","con","para","por",
    "que","del","las","los","una","uno","todo","pero","este","esta",
    "ese","esa","hay","tan","solo","cuando","como","porque","aunque",
    "sino","cada","vez","algo","poco","mucho","demasiado","él","ella",
    "me","te","se","nos","le","lo","la","su","sus","mi","mis","tu",
}
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    try:
        tokens = word_tokenize(text, language="spanish")
    except Exception:
        tokens = text.split()
    return [lemmatizer.lemmatize(t) for t in tokens
            if t not in STOP_ES and len(t) > 2]

def classify_sentiment(text, client):
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content":
                    "Eres un clasificador de sentimientos en español. "
                    "Responde ÚNICAMENTE con una de estas palabras: Positivo, Negativo, Neutro."},
                {"role": "user", "content": f"Clasifica: {text}"}
            ],
            max_tokens=10, temperature=0
        )
        result = r.choices[0].message.content.strip()
        for lbl in ["Positivo", "Negativo", "Neutro"]:
            if lbl.lower() in result.lower():
                return lbl
        return "Neutro"
    except Exception as e:
        st.session_state["groq_error"] = str(e)
        return "Error"

# ── Paleta dark azul-morado ────────────────────────────────────────
C_BG        = "#0f0f1a"   # fondo principal
C_CARD      = "#1a1a2e"   # tarjetas / secciones
C_BORDER    = "#2d2b55"   # bordes
C_ACCENT1   = "#7c3aed"   # morado principal
C_ACCENT2   = "#3b82f6"   # azul
C_ACCENT3   = "#a78bfa"   # morado claro
C_TEXT      = "#e2e8f0"   # texto principal
C_SUBTEXT   = "#94a3b8"   # texto secundario
PLOTLY_BG   = "#1a1a2e"
PLOTLY_GRID = "#2d2b55"

COLORSCALE  = [[0, "#3b82f6"], [0.5, "#7c3aed"], [1, "#a78bfa"]]
DISC_COLORS = ["#7c3aed", "#3b82f6", "#a78bfa", "#60a5fa", "#c4b5fd"]

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(page_title="Análisis NLP", page_icon="🧠", layout="wide")

st.markdown(f"""
<style>
    /* Fondo general */
    .stApp {{ background-color: {C_BG}; color: {C_TEXT}; }}
    [data-testid="stAppViewContainer"] {{ background-color: {C_BG}; }}
    [data-testid="stSidebar"] {{ background-color: {C_CARD}; border-right: 1px solid {C_BORDER}; }}

    /* Encabezado principal */
    .nlp-header {{
        background: linear-gradient(135deg, #1e1b4b 0%, #2d1b69 50%, #1e3a5f 100%);
        border: 1px solid {C_BORDER};
        border-radius: 14px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .nlp-header h1 {{ color: {C_ACCENT3}; font-size: 2rem; margin: 0; }}
    .nlp-header p  {{ color: {C_SUBTEXT}; margin: .4rem 0 0; font-size: .95rem; }}

    /* Tarjetas de sección */
    .section-card {{
        background: {C_CARD};
        border: 1px solid {C_BORDER};
        border-left: 4px solid {C_ACCENT1};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0 .5rem;
    }}
    .section-card h3 {{ color: {C_ACCENT3}; margin: 0 0 .2rem; font-size: 1.05rem; }}
    .section-card p  {{ color: {C_SUBTEXT}; margin: 0; font-size: .85rem; }}

    /* Métricas */
    [data-testid="stMetric"] {{
        background: {C_CARD};
        border: 1px solid {C_BORDER};
        border-radius: 10px;
        padding: .8rem 1rem;
    }}
    [data-testid="stMetricLabel"]  {{ color: {C_SUBTEXT} !important; }}
    [data-testid="stMetricValue"]  {{ color: {C_ACCENT3} !important; font-size: 1.6rem !important; }}

    /* Botones */
    .stButton > button {{
        background: linear-gradient(135deg, {C_ACCENT1}, {C_ACCENT2});
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: .5rem 1.5rem;
        transition: opacity .2s;
    }}
    .stButton > button:hover {{ opacity: .85; }}

    /* Selectbox y file uploader */
    .stSelectbox > div > div {{
        background: {C_CARD} !important;
        border: 1px solid {C_BORDER} !important;
        color: {C_TEXT} !important;
        border-radius: 8px !important;
    }}

    /* Divider */
    hr {{ border-color: {C_BORDER} !important; }}

    /* Caption */
    .stCaption {{ color: {C_SUBTEXT} !important; }}

    /* Text area */
    textarea {{
        background: {C_CARD} !important;
        color: {C_TEXT} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 8px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="nlp-header">
    <h1>🧠 Análisis NLP de Opiniones</h1>
    <p>Procesamiento de Lenguaje Natural · Nube de Palabras · Clasificación con IA · Groq LLaMA 3.1</p>
</div>
""", unsafe_allow_html=True)

# ── Helper para títulos de sección ────────────────────────────────
def section(num, titulo, subtitulo=""):
    st.markdown(f"""
    <div class="section-card">
        <h3>{num}. {titulo}</h3>
        {"<p>" + subtitulo + "</p>" if subtitulo else ""}
    </div>
    """, unsafe_allow_html=True)

# ── Cargar CSV ────────────────────────────────────────────────────
@st.cache_data
def load_default():
    return pd.read_csv("Comentarios.csv", sep=";", encoding="utf-8-sig")

def label_rating(r):
    if r <= 2:   return "Negativa"
    elif r == 3: return "Neutral"
    else:        return "Positiva"

section("1", "Cargar opiniones", "Sube tu CSV o se usará la base de datos por defecto")
uploaded = st.file_uploader("Archivo CSV (separador ; o ,)", type=["csv"], label_visibility="collapsed")

if uploaded:
    sample = uploaded.read(2048).decode("utf-8-sig")
    uploaded.seek(0)
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(uploaded, sep=sep, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    st.success(f"✅ {len(df)} opiniones cargadas")
else:
    df = load_default()
    st.info("📂 Usando Comentarios.csv por defecto")

df["Comentario"] = df["Comentario"].astype(str)
df["Ratings"]    = pd.to_numeric(df["Ratings"], errors="coerce").fillna(3).astype(int)
df["Clase"]      = df["Ratings"].apply(label_rating)

# Métricas rápidas
c1, c2, c3 = st.columns(3)
c1.metric("📝 Total opiniones", len(df))
c2.metric("⭐ Rating promedio", f"{df['Ratings'].mean():.1f}")
c3.metric("💬 Palabras promedio", f"{df['Comentario'].apply(lambda x: len(x.split())).mean():.0f}")

# ── Filtro ────────────────────────────────────────────────────────
st.divider()
section("2", "Filtro por clase", "Procesa solo las opiniones de una categoría específica")

clases    = ["Todas"] + sorted(df["Clase"].unique().tolist())
clase_sel = st.selectbox("Mostrar clase:", clases)
df_f = df if clase_sel == "Todas" else df[df["Clase"] == clase_sel]

st.dataframe(
    df_f[["ID", "Comentario", "Ratings", "Clase"]],
    use_container_width=True, height=220
)

# ── Nube de palabras + Top 10 ─────────────────────────────────────
st.divider()
section("3", "Frecuencia de palabras", "Stopwords eliminadas · Lematización aplicada con NLTK")

tokens_all = []
for t in df_f["Comentario"]:
    tokens_all.extend(preprocess(t))

if tokens_all:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<p style='color:{C_SUBTEXT};font-weight:600;margin-bottom:.5rem'>☁️ Nube de palabras</p>", unsafe_allow_html=True)
        WC_PAL = ["#7c3aed","#3b82f6","#a78bfa","#60a5fa",
                  "#c4b5fd","#818cf8","#6366f1","#93c5fd"]
        def wc_color(word, **kwargs):
            return WC_PAL[hash(word) % len(WC_PAL)]
        wc = WordCloud(
            width=700, height=380,
            background_color="#1a1a2e",
            color_func=wc_color,
            max_words=60,
            prefer_horizontal=0.85
        ).generate(" ".join(tokens_all))
        fig_wc, ax = plt.subplots(figsize=(7, 3.8))
        fig_wc.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
        plt.close(fig_wc)

    with col2:
        st.markdown(f"<p style='color:{C_SUBTEXT};font-weight:600;margin-bottom:.5rem'>📊 Top 10 palabras más frecuentes</p>", unsafe_allow_html=True)
        top = Counter(tokens_all).most_common(10)
        words, counts = zip(*top)
        fig_bar = px.bar(
            x=list(counts), y=list(words), orientation="h",
            labels={"x": "Frecuencia", "y": ""},
            color=list(counts),
            color_continuous_scale=COLORSCALE
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed", color=C_TEXT),
            xaxis=dict(color=C_TEXT),
            coloraxis_showscale=False,
            plot_bgcolor=PLOTLY_BG,
            paper_bgcolor=PLOTLY_BG,
            font=dict(color=C_TEXT),
            height=380,
            margin=dict(l=10, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning("No hay tokens suficientes con el filtro actual.")

# ── Gráfico adicional ─────────────────────────────────────────────
st.divider()
section("4", "Análisis de longitud de opiniones", "Distribución de palabras por opinión según su clase")

df_f2 = df_f.copy()
df_f2["n_palabras"] = df_f2["Comentario"].apply(lambda x: len(x.split()))

fig_box = px.box(
    df_f2, x="Clase", y="n_palabras", color="Clase",
    labels={"n_palabras": "Palabras por opinión", "Clase": ""},
    color_discrete_sequence=DISC_COLORS,
    points="all"
)
fig_box.update_layout(
    showlegend=False,
    plot_bgcolor=PLOTLY_BG,
    paper_bgcolor=PLOTLY_BG,
    font=dict(color=C_TEXT),
    xaxis=dict(color=C_TEXT, gridcolor=PLOTLY_GRID),
    yaxis=dict(color=C_TEXT, gridcolor=PLOTLY_GRID),
    height=360
)
st.plotly_chart(fig_box, use_container_width=True)

# ── Clasificación LLM ─────────────────────────────────────────────
st.divider()
section("5", "Clasificación de sentimientos con LLM", "Groq · LLaMA 3.1 · Positivo / Negativo / Neutro")

if st.button("🚀 Clasificar opiniones", type="primary"):
    client  = get_client()
    results = []
    progress = st.progress(0, text="Clasificando…")
    for i, row in enumerate(df_f.itertuples()):
        sentiment = classify_sentiment(row.Comentario, client)
        results.append({
            "ID": getattr(row, "ID", i + 1),
            "Opinión": row.Comentario,
            "Rating": row.Ratings,
            "Clase real": row.Clase,
            "Sentimiento LLM": sentiment
        })
        progress.progress((i + 1) / len(df_f), text=f"Clasificando {i+1}/{len(df_f)}…")
    progress.empty()
    st.session_state["df_res"] = pd.DataFrame(results)

if "groq_error" in st.session_state:
    st.error(f"❌ Error Groq: {st.session_state['groq_error']}")

if "df_res" in st.session_state:
    df_res = st.session_state["df_res"]

    color_map_td = {
        "Positivo": "background-color:#14532d;color:#86efac",
        "Negativo": "background-color:#4c0519;color:#fca5a5",
        "Neutro":   "background-color:#3b2f00;color:#fcd34d",
        "Error":    "background-color:#1e1e2e;color:#94a3b8"
    }
    def highlight(val):
        return color_map_td.get(val, "")

    st.markdown(f"<p style='color:{C_SUBTEXT};font-weight:600;margin:.8rem 0 .3rem'>📋 Resultados por opinión</p>", unsafe_allow_html=True)
    st.dataframe(
        df_res.style.map(highlight, subset=["Sentimiento LLM"]),
        use_container_width=True, height=360
    )

    dist = df_res["Sentimiento LLM"].value_counts().reset_index()
    dist.columns = ["Sentimiento", "Cantidad"]
    dist["Porcentaje"] = (dist["Cantidad"] / dist["Cantidad"].sum() * 100).round(1)

    color_map_fig = {
        "Positivo": "#22c55e",
        "Negativo": "#ef4444",
        "Neutro":   "#f59e0b",
        "Error":    "#6b7280"
    }
    fig_sent = px.pie(
        dist, values="Cantidad", names="Sentimiento",
        color="Sentimiento", color_discrete_map=color_map_fig,
        hole=0.5,
        title="Distribución de sentimientos"
    )
    fig_sent.update_traces(textinfo="percent+label", textfont_color="white")
    fig_sent.update_layout(
        showlegend=True,
        legend=dict(font=dict(color=C_TEXT)),
        title_font_color=C_ACCENT3,
        paper_bgcolor=PLOTLY_BG,
        font=dict(color=C_TEXT),
        height=400
    )
    st.plotly_chart(fig_sent, use_container_width=True)

# ── Nuevo comentario ──────────────────────────────────────────────
st.divider()
section("6", "Analizar un nuevo comentario", "Escribe cualquier opinión y el modelo la clasificará en tiempo real")

nuevo = st.text_area("Tu comentario:", placeholder="Escribe aquí una opinión…", height=110, label_visibility="collapsed")
if st.button("🔍 Analizar comentario", type="primary"):
    if not nuevo.strip():
        st.warning("Escribe un comentario antes de analizar.")
    else:
        with st.spinner("Analizando con LLaMA 3.1…"):
            sentiment = classify_sentiment(nuevo, get_client())
        configs = {
            "Positivo": ("#22c55e", "#052e16", "😊"),
            "Negativo": ("#ef4444", "#2d0a0a", "😠"),
            "Neutro":   ("#f59e0b", "#2d1f00", "😐"),
        }
        color, bg, emoji = configs.get(sentiment, ("#6b7280", "#1e1e2e", "🤔"))
        st.markdown(f"""
        <div style="padding:1.4rem 1.8rem;border-radius:12px;
                    border-left:5px solid {color};
                    background:{bg};margin-top:.8rem">
            <h4 style="color:{color};margin:0 0 .5rem">
                {emoji} Sentimiento detectado: <strong>{sentiment}</strong>
            </h4>
            <p style="color:#cbd5e1;margin:0;font-style:italic;">
                "{nuevo}"
            </p>
        </div>
        """, unsafe_allow_html=True)