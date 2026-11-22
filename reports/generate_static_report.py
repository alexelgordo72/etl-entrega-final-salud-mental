import os
import json
import base64
from io import BytesIO
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

DB_URI = "postgresql+psycopg2://airflow:airflow@localhost:5433/suicidios_colombia"
REPORT_PATH = "/home/alexelgordo/etl_entrega_ii/reports/reporte_analitico_estatico.html"
METRICS_PATH = "/home/alexelgordo/etl_entrega_ii/models/model_metrics.json"

sns.set_theme(style="whitegrid")

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    engine = create_engine(DB_URI)
    df = pd.read_sql("SELECT * FROM public.fact_suicidios_enriquecida_v2", engine)
    cols_base = [c for c in df.columns if not c.endswith("_x") and not c.endswith("_y")]
    df = df[cols_base].copy()

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    # 1. Evolución anual vs Tasa de Desempleo (World Bank API)
    anual = df.groupby("anio").agg(casos=("cantidad", "sum"), desempleo=("tasa_desempleo", "mean")).reset_index()
    fig1, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.bar(anual["anio"], anual["casos"], color="#457B9D", alpha=0.85, label="Presuntos Suicidios")
    ax1.set_ylabel("Total de Casos", color="#457B9D")
    ax1.set_title("Evolución Anual de Presuntos Suicidios vs. Tasa de Desempleo (Colombia)")
    ax2 = ax1.twinx()
    ax2.plot(anual["anio"], anual["desempleo"], color="#E63946", marker="o", linewidth=2.2, label="Desempleo (%)")
    ax2.set_ylabel("Tasa de Desempleo (%)", color="#E63946")
    img1 = fig_to_base64(fig1)

    # 2. Top 10 Departamentos
    top_dep = df.groupby("departamento")["cantidad"].sum().sort_values(ascending=False).head(10).reset_index()
    fig2, ax = plt.subplots(figsize=(9, 4.2))
    sns.barplot(data=top_dep, x="cantidad", y="departamento", palette="Reds_r", ax=ax)
    ax.set_title("Top 10 Departamentos con Mayor Concentración Histórica de Casos")
    ax.set_xlabel("Total de Casos")
    ax.set_ylabel("")
    img2 = fig_to_base64(fig2)

    # 3. Zoom Valle del Cauca (Tasa por 100k hab usando Scraping Wikipedia)
    valle = df[df["departamento"].str.upper().str.contains("VALLE", na=False)].copy()
    valle_agg = valle.groupby("municipio").agg(casos=("cantidad", "sum"), pob=("poblacion_2018", "first")).dropna().reset_index()
    valle_agg = valle_agg[valle_agg["pob"] > 0]
    valle_agg["tasa_100k"] = (valle_agg["casos"] / valle_agg["pob"]) * 100000
    top_valle = valle_agg.sort_values("tasa_100k", ascending=False).head(10)

    fig3, ax = plt.subplots(figsize=(9, 4.2))
    sns.barplot(data=top_valle, x="tasa_100k", y="municipio", palette="viridis", ax=ax)
    ax.set_title("Valle del Cauca: Top 10 Municipios por Tasa Acumulada por 100.000 Habitantes")
    ax.set_xlabel("Tasa acumulada por 100.000 habitantes")
    ax.set_ylabel("")
    img3 = fig_to_base64(fig3)

    anio_pico = int(anual.loc[anual["casos"].idxmax(), "anio"])
    casos_pico = int(anual["casos"].max())
    depto_top = top_dep.iloc[0]["departamento"]
    muni_tasa_top = top_valle.iloc[0]["municipio"] if not top_valle.empty else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Reporte Analítico Estático - ODS 3 Salud Mental</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px auto; max-width: 960px; color: #222; line-height: 1.6; }}
  h1, h2 {{ color: #1D3557; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 20px 0; }}
  .kpi {{ background: #F1FAEE; border-left: 4px solid #457B9D; padding: 14px; border-radius: 6px; }}
  .kpi strong {{ font-size: 1.4em; display: block; color: #1D3557; }}
  .insight {{ background: #FFF8E7; border-left: 4px solid #E9C46A; padding: 14px; margin: 16px 0; }}
  img {{ width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 12px 0; }}
  footer {{ margin-top: 40px; font-size: 0.85em; color: #666; border-top: 1px solid #ddd; padding-top: 10px; }}
</style>
</head>
<body>
  <h1>Reporte Analítico Estático: Presuntos Suicidios en Colombia (ODS 3)</h1>
  <p><strong>Generado:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} | <strong>Fuente:</strong> Data Warehouse PostgreSQL (<code>fact_suicidios_enriquecida_v2</code>)</p>

  <div class="kpi-grid">
    <div class="kpi"><span>Registros Validados</span><strong>{len(df):,}</strong></div>
    <div class="kpi"><span>Año Pico</span><strong>{anio_pico} ({casos_pico:,})</strong></div>
    <div class="kpi"><span>Depto. Mayor Volumen</span><strong>{depto_top}</strong></div>
    <div class="kpi"><span>ROC-AUC Modelo ML</span><strong>{metrics['roc_auc']}</strong></div>
  </div>

  <h2>1. Dinámica Temporal y Contexto Macroeconómico</h2>
  <img src="data:image/png;base64,{img1}" alt="Evolución anual">
  <div class="insight">
    <strong>Insight:</strong> El volumen anual alcanzó su punto máximo en <strong>{anio_pico}</strong> con <strong>{casos_pico:,}</strong> registros. El cruce con la API del Banco Mundial evidencia que el repunte post-pandemia coincide con presiones sostenidas en el mercado laboral.
  </div>

  <h2>2. Distribución Geográfica Nacional</h2>
  <img src="data:image/png;base64,{img2}" alt="Top departamentos">
  <div class="insight">
    <strong>Insight:</strong> <strong>{depto_top}</strong> lidera el volumen absoluto histórico, seguido por los principales centros urbanos del país, lo que justifica focalizar estrategias diferenciadas entre cabeceras municipales y zonas rurales.
  </div>

  <h2>3. Zoom Territorial: Valle del Cauca (Tasa por 100.000 Habitantes)</h2>
  <img src="data:image/png;base64,{img3}" alt="Tasa Valle del Cauca">
  <div class="insight">
    <strong>Insight:</strong> Al normalizar los casos con los datos poblacionales extraídos por web scraping, municipios intermedios como <strong>{muni_tasa_top}</strong> revelan una incidencia proporcionalmente alta que queda oculta cuando se analizan únicamente conteos absolutos.
  </div>

  <h2>4. Desempeño del Modelo Predictivo (Flujo Operacional)</h2>
  <p>El modelo <strong>Random Forest Classifier</strong> entrenado con división temporal (entrenamiento &le; 2022, prueba &ge; 2023) obtuvo un <strong>ROC-AUC de {metrics['roc_auc']}</strong> y un <strong>F1-Score de {metrics['f1_score']}</strong> sobre {metrics['test_rows']:,} registros de prueba, priorizando un alto <em>Recall</em> para la detección temprana de municipios en riesgo alto vía Kafka.</p>

  <footer>Generado automáticamente en Python desde el modelo dimensional validado con Great Expectations y Apache Airflow.</footer>
</body>
</html>"""

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"ÉXITO: Reporte estático generado en {REPORT_PATH}")

if __name__ == "__main__":
    main()
