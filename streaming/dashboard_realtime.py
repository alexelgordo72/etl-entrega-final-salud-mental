import time
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(
    page_title="Alerta Temprana Salud Mental - Streaming Kafka",
    layout="wide",
)

DB_URI = "postgresql+psycopg2://airflow:airflow@localhost:5433/suicidios_colombia"

@st.cache_resource
def get_engine():
    return create_engine(DB_URI)

def load_predictions():
    engine = get_engine()
    query = """
        SELECT event_id, event_timestamp, anio, mes, departamento, municipio,
               zona_hecho, tasa_desempleo, prediccion_riesgo, etiqueta_riesgo,
               probabilidad_riesgo, processed_at
        FROM public.predicciones_tiempo_real
        ORDER BY id DESC
        LIMIT 500
    """
    return pd.read_sql(query, engine)

st.title("Sistema de Alerta Temprana en Tiempo Real (Kafka + ML)")
st.caption("Flujo operacional alimentado por el consumidor de Kafka (`suicidios-riesgo-stream`) e inferencia con Random Forest (`.pkl`).")

col_ctrl1, col_ctrl2 = st.columns([1, 4])
with col_ctrl1:
    auto_refresh = st.toggle("Auto-actualizar (3s)", value=True)
with col_ctrl2:
    if st.button("Actualizar ahora"):
        st.rerun()

try:
    df = load_predictions()
except Exception as e:
    st.error(f"Error conectando a PostgreSQL: {e}")
    st.stop()

if df.empty:
    st.warning("Aún no hay predicciones en la tabla. Ejecuta el productor de Kafka para ver el flujo en vivo.")
else:
    total = len(df)
    altos = int((df["prediccion_riesgo"] == 1).sum())
    pct_alto = (altos / total) * 100
    prob_prom = df["probabilidad_riesgo"].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Eventos Procesados", f"{total:,}")
    k2.metric("Alertas Riesgo Alto", f"{altos:,}", delta=f"{pct_alto:.1f}% del flujo", delta_color="inverse")
    k3.metric("Riesgo Moderado/Bajo", f"{total - altos:,}")
    k4.metric("Probabilidad Promedio", f"{prob_prom:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(
            df,
            names="etiqueta_riesgo",
            title="Distribución de Predicciones en Vivo",
            color="etiqueta_riesgo",
            color_discrete_map={"Riesgo Alto": "#E63946", "Moderado/Bajo": "#2A9D8F"},
            hole=0.45,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        top_deptos = (
            df[df["prediccion_riesgo"] == 1]
            .groupby("departamento")
            .size()
            .reset_index(name="alertas")
            .sort_values("alertas", ascending=False)
            .head(10)
        )
        fig_bar = px.bar(
            top_deptos,
            x="alertas",
            y="departamento",
            orientation="h",
            title="Top Departamentos con Alertas de Riesgo Alto",
            color="alertas",
            color_continuous_scale="Reds",
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Últimas Predicciones Recibidas desde Kafka")
    st.dataframe(
        df[["event_id", "departamento", "municipio", "mes", "zona_hecho", "tasa_desempleo", "etiqueta_riesgo", "probabilidad_riesgo", "processed_at"]].head(25),
        use_container_width=True,
    )

if auto_refresh:
    time.sleep(3)
    st.rerun()
