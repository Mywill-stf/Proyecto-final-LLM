# ╔══════════════════════════════════════════════════════════════════╗
# ║        Análisis NLP de Opiniones de Clientes                    ║
# ║        Streamlit App · Groq (LLaMA 3) · NLTK                   ║
# ╚══════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from openai import OpenAI

# ── NLTK ──────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)

download_nltk()

# ── Groq client ───────────────────────────────────────────────────
GROQ_API_KEY = "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"
GROQ_MODEL   = "llama3-8b-8192"

@st.cache_resource
def get_client():
    return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# ── NLP helpers ───────────────────────────────────────────────────
STOP_ES = set(stopwords.words("spanish")) | {
    "más","si","así","aquí","ahí","ser","estar","hacer","tener","ir",
    "ver","muy","bien","mal","ya","también","sin","con","para","por",
    "que","del","las","los","una","uno","todo","pero","este","esta",
    "ese","esa","hay","tan","solo","solo","cuando","como","porque",
    "aunque","sino","cada","vez","algo","poco","mucho","demasiado",
    "él","ella","ellos","nosotros","vosotros","me","te","se","nos",
    "le","lo","la","su","sus","mi","mis","tu","tus","sus","nuestro",
}
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    try:
        tokens = word_tokenize(text, language="spanish")
    except Exception:
        tokens = text.split()
    return [lemmatizer.lemmatize(t) for t in tokens
            if t not in STOP_ES and len(t) > 2]

def top_words(series: pd.Series, n=10):
    all_tok = []
    for t in series:
        all_tok.extend(preprocess(str(t)))
    return Counter(all_tok).most_common(n)

# ── Sentiment via LLM ─────────────────────────────────────────────
def classify_sentiment(text: str, client) -> str:
    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": ("Eres un clasificador de sentimientos en español. "
                              "Responde ÚNICAMENTE con una de estas tres palabras: "
                              "Positivo, Negativo, Neutro. Sin más texto.")},
                {"role": "user",
                 "content": f"Clasifica el sentimiento: {text}"}
            ],
            max_tokens=10, temperature=0
        )
        result = r.choices[0].message.content.strip()
        for lbl in ["Positivo", "Negativo", "Neutro"]:
            if lbl.lower() in result.lower():
                return lbl
        return "Neutro"
    except Exception as e:
        return "Error"

# ── Cargar datos ──────────────────────────────────────────────────
@st.cache_data
def load_default():
    return pd.read_csv("Comentarios.csv", sep=";", encoding="utf-8-sig")

def label_rating(r):
    if r <= 2:   return "⭐ Negativa (1-2)"
    elif r == 3: return "⭐⭐ Neutral (3)"
    else:        return "⭐⭐⭐ Positiva (4-5)"

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NLP Opiniones",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #2980b9;
        padding: 1rem;
        border-radius: 8px;
    }
    .stTab [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧠 Análisis NLP de Opiniones de Clientes</h1>
    <p>Procesamiento de Lenguaje Natural · Nube de Palabras · Clasificación con IA (Groq + LLaMA 3)</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuración")
    st.divider()

    st.subheader("📂 Datos")
    uploaded = st.file_uploader("Sube tu CSV (sep=; o ,)", type=["csv"])

    if uploaded:
        try:
            # Detectar separador
            sample = uploaded.read(2048).decode("utf-8-sig")
            uploaded.seek(0)
            sep = ";" if sample.count(";") > sample.count(",") else ","
            df = pd.read_csv(uploaded, sep=sep, encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            # Detectar columnas
            col_comment = next((c for c in df.columns if "coment" in c.lower() or "opinion" in c.lower() or "review" in c.lower()), df.columns[1])
            col_rating  = next((c for c in df.columns if "rating" in c.lower() or "stars" in c.lower() or "estrella" in c.lower() or "clase" in c.lower()), df.columns[2])
            df = df.rename(columns={col_comment: "Comentario", col_rating: "Ratings"})
            st.success(f"✅ {len(df)} opiniones cargadas")
        except Exception as e:
            st.error(f"Error: {e}")
            df = load_default()
    else:
        df = load_default()
        st.info("📌 Usando base de datos predeterminada (Comentarios.csv)")

    df["Comentario"] = df["Comentario"].astype(str)
    df["Ratings"]    = pd.to_numeric(df["Ratings"], errors="coerce").fillna(3).astype(int)
    df["Clase"]      = df["Ratings"].apply(label_rating)

    st.divider()
    st.subheader("🔍 Filtro por clase")
    clases = ["Todas"] + sorted(df["Clase"].unique().tolist())
    clase_sel = st.selectbox("Seleccionar:", clases)

    st.divider()
    st.markdown(f"**Total opiniones:** {len(df)}")
    st.markdown(f"**Clases:** {df['Clase'].nunique()}")
    st.markdown(f"**Filtro activo:** {clase_sel}")

# ── Filtrado ──────────────────────────────────────────────────────
df_f = df if clase_sel == "Todas" else df[df["Clase"] == clase_sel]

# ── MÉTRICAS RÁPIDAS ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📝 Opiniones", len(df_f))
c2.metric("⭐ Rating promedio", f"{df_f['Ratings'].mean():.1f}")
c3.metric("📊 Clases", df_f["Clase"].nunique())
avg_words = df_f["Comentario"].apply(lambda x: len(x.split())).mean()
c4.metric("💬 Palabras/opinión", f"{avg_words:.0f}")

st.divider()

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "☁️ Nube & Top palabras",
    "📊 Análisis adicional",
    "🤖 Clasificación LLM",
    "💬 Nuevo comentario"
])

# ── TAB 1 ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("☁️ Análisis de frecuencia de palabras")
    st.caption("Stopwords eliminadas · Lematización aplicada con WordNetLemmatizer (NLTK)")

    tokens_all = []
    for t in df_f["Comentario"]:
        tokens_all.extend(preprocess(t))

    if not tokens_all:
        st.warning("No hay suficientes tokens para graficar con el filtro actual.")
    else:
        text_joined = " ".join(tokens_all)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("#### ☁️ Nube de palabras")
            wc = WordCloud(
                width=700, height=420,
                background_color="white",
                colormap="Blues",
                max_words=60,
                prefer_horizontal=0.85
            ).generate(text_joined)
            fig_wc, ax_wc = plt.subplots(figsize=(7, 4.2))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            fig_wc.patch.set_facecolor("white")
            st.pyplot(fig_wc)
            plt.close(fig_wc)

        with col2:
            st.markdown("#### 📊 Top 10 palabras más frecuentes")
            top = top_words(df_f["Comentario"], 10)
            if top:
                words, counts = zip(*top)
                colors = [f"hsl({210 + i*8}, 70%, {55 - i*3}%)" for i in range(len(words))]
                fig_bar = go.Figure(go.Bar(
                    x=list(counts),
                    y=list(words),
                    orientation="h",
                    marker_color=colors,
                    text=list(counts),
                    textposition="outside"
                ))
                fig_bar.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Frecuencia",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=380,
                    margin=dict(l=10, r=40, t=20, b=30)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(f"**Total tokens únicos procesados:** {len(set(tokens_all))} | **Total ocurrencias:** {len(tokens_all)}")

# ── TAB 2 ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Análisis adicional de texto")

    df_f2 = df_f.copy()
    df_f2["n_palabras"]  = df_f2["Comentario"].apply(lambda x: len(x.split()))
    df_f2["n_tokens"]    = df_f2["Comentario"].apply(lambda x: len(preprocess(x)))
    df_f2["n_caracteres"]= df_f2["Comentario"].apply(len)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📏 Distribución de longitud por clase")
        fig_box = px.box(
            df_f2, x="Clase", y="n_palabras",
            color="Clase",
            labels={"n_palabras": "Palabras por opinión", "Clase": ""},
            color_discrete_sequence=px.colors.qualitative.Set2,
            points="all"
        )
        fig_box.update_layout(showlegend=False, plot_bgcolor="white", height=360)
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.markdown("#### 🥧 Distribución de clases (Ratings)")
        conteo = df_f2["Clase"].value_counts().reset_index()
        conteo.columns = ["Clase", "Cantidad"]
        fig_pie = px.pie(
            conteo, values="Cantidad", names="Clase",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 📈 Palabras promedio por rating (1–5 ⭐)")
    avg_por_rating = df_f2.groupby("Ratings")["n_palabras"].mean().reset_index()
    avg_por_rating.columns = ["Rating", "Promedio palabras"]
    fig_line = px.bar(
        avg_por_rating, x="Rating", y="Promedio palabras",
        color="Rating",
        color_continuous_scale="Blues",
        labels={"Rating": "Estrellas", "Promedio palabras": "Promedio de palabras"},
        text_auto=".1f"
    )
    fig_line.update_layout(plot_bgcolor="white", coloraxis_showscale=False, height=300)
    st.plotly_chart(fig_line, use_container_width=True)

    with st.expander("🔢 Ver estadísticas detalladas"):
        st.dataframe(
            df_f2.groupby("Clase")[["n_palabras","n_tokens","n_caracteres"]]
                  .describe().round(1),
            use_container_width=True
        )

# ── TAB 3 ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 Clasificación de sentimientos con LLaMA 3 via Groq")
    st.markdown(
        "El modelo analiza cada opinión de la selección actual y la clasifica como "
        "**Positivo**, **Negativo** o **Neutro**."
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_llm = st.button("🚀 Clasificar opiniones", type="primary", use_container_width=True)
    with col_info:
        st.info(f"Se clasificarán **{len(df_f)}** opiniones · Modelo: `{GROQ_MODEL}`")

    if run_llm:
        client = get_client()
        results = []
        progress = st.progress(0, text="Iniciando clasificación…")
        status   = st.empty()

        for i, row in enumerate(df_f.itertuples()):
            sentiment = classify_sentiment(row.Comentario, client)
            results.append({
                "ID":           getattr(row, "ID", i + 1),
                "Opinión":      row.Comentario,
                "Rating real":  row.Ratings,
                "Clase real":   row.Clase,
                "Sentimiento":  sentiment
            })
            progress.progress((i + 1) / len(df_f), text=f"Clasificando {i+1}/{len(df_f)}…")
            status.caption(f"✔ [{sentiment}] {row.Comentario[:60]}…")

        progress.empty()
        status.empty()
        st.session_state["df_res"] = pd.DataFrame(results)
        st.success("✅ Clasificación completada")

    if "df_res" in st.session_state:
        df_res = st.session_state["df_res"]

        # Tabla con colores
        def color_sent(val):
            colors = {"Positivo": "#d4edda", "Negativo": "#f8d7da", "Neutro": "#fff3cd", "Error": "#e2e3e5"}
            return f"background-color:{colors.get(val,'white')}"

        st.markdown("#### 📋 Resultados por opinión")
        st.dataframe(
            df_res.style.applymap(color_sent, subset=["Sentimiento"]),
            use_container_width=True,
            height=400
        )

        # Gráfico porcentajes
        st.markdown("#### 📊 Distribución de sentimientos detectados")
        dist = df_res["Sentimiento"].value_counts().reset_index()
        dist.columns = ["Sentimiento", "Cantidad"]
        dist["Porcentaje"] = (dist["Cantidad"] / dist["Cantidad"].sum() * 100).round(1)

        color_map = {"Positivo": "#28a745", "Negativo": "#dc3545", "Neutro": "#ffc107", "Error": "#6c757d"}

        col1, col2 = st.columns(2)
        with col1:
            fig_sent_pie = px.pie(
                dist, values="Cantidad", names="Sentimiento",
                color="Sentimiento", color_discrete_map=color_map,
                hole=0.45
            )
            fig_sent_pie.update_traces(textinfo="percent+label")
            fig_sent_pie.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_sent_pie, use_container_width=True)

        with col2:
            fig_sent_bar = px.bar(
                dist, x="Sentimiento", y="Cantidad",
                color="Sentimiento", color_discrete_map=color_map,
                text=dist["Porcentaje"].astype(str) + "%"
            )
            fig_sent_bar.update_layout(showlegend=False, plot_bgcolor="white", height=320)
            st.plotly_chart(fig_sent_bar, use_container_width=True)

        # Métricas rápidas
        st.markdown("#### 📌 Resumen")
        c1, c2, c3 = st.columns(3)
        for col, label, emoji in zip([c1, c2, c3],
                                      ["Positivo", "Negativo", "Neutro"],
                                      ["😊", "😠", "😐"]):
            n = dist.loc[dist["Sentimiento"] == label, "Cantidad"].values
            pct = dist.loc[dist["Sentimiento"] == label, "Porcentaje"].values
            col.metric(f"{emoji} {label}", f"{n[0] if len(n) else 0}", f"{pct[0] if len(pct) else 0}%")

# ── TAB 4 ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("💬 Analiza un nuevo comentario")
    st.markdown("Escribe cualquier opinión y el modelo la clasificará en tiempo real.")

    nuevo = st.text_area(
        "✏️ Tu comentario:",
        placeholder="Escribe aquí una opinión de cliente…",
        height=120
    )

    if st.button("🔍 Analizar comentario", type="primary"):
        if not nuevo.strip():
            st.warning("Por favor escribe un comentario antes de analizar.")
        else:
            with st.spinner("Analizando con LLaMA 3…"):
                client    = get_client()
                sentiment = classify_sentiment(nuevo, client)

            emojis = {"Positivo": "😊", "Negativo": "😠", "Neutro": "😐"}
            colors = {"Positivo": "green", "Negativo": "red", "Neutro": "orange"}
            emoji  = emojis.get(sentiment, "🤔")
            color  = colors.get(sentiment, "gray")

            st.markdown(f"""
            <div style="padding:1.5rem;border-radius:12px;border-left:6px solid {color};
                        background:#f8f9fa;margin-top:1rem;">
                <h3 style="color:{color};margin:0">{emoji} Sentimiento detectado: <strong>{sentiment}</strong></h3>
                <p style="color:#555;margin-top:.5rem;font-style:italic;">"{nuevo}"</p>
            </div>
            """, unsafe_allow_html=True)

            # Mini análisis de tokens
            tokens = preprocess(nuevo)
            if tokens:
                st.markdown(f"**Tokens relevantes:** `{' · '.join(tokens[:15])}`")

    st.divider()
    st.caption("Modelo: LLaMA 3 (8B) vía API de Groq · NLP: NLTK · Visualizaciones: Plotly")

