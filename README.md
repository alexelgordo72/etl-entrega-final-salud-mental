# Arquitectura Híbrida ETL + ML Streaming: Salud Mental en Colombia (ODS 3)

Proyecto final del curso **ETL (G51)** — Ingeniería de Datos e Inteligencia Artificial, Universidad Autónoma de Occidente.

## 1. Diagrama de Arquitectura
* **Fuentes:** PostgreSQL (`suicidios_colombia`), CSV DANE/GCS, APIs World Bank y Nager.Date, Web Scraping Wikipedia.
* **Orquestación (Airflow):** `Extract (5 tareas)` -> `Transform & Merge` -> `Great Expectations` -> `Load DW + Cuarentena` -> `Kafka Producer`.
* **Consumo Analítico (Batch):** `PostgreSQL DW (fact_suicidios_enriquecida_v2)` -> `Reporte Analítico Estático (Python -> HTML)`.
* **Consumo Operacional (Tiempo Real):** `Kafka (suicidios-riesgo-stream)` -> `Kafka Consumer + RandomForest (.pkl)` -> `PostgreSQL (predicciones_tiempo_real)` -> `Dashboard Streamlit`.

## 2. Articulación de Valor
* **Analítico (Batch):** Consolida y valida registros históricos (2015–2024) con desempleo (Banco Mundial), festivos y datos territoriales para evaluar tendencias y tasas por 100.000 habitantes en el reporte estático.
* **Operacional (Tiempo Real):** Evalúa eventos territoriales en streaming vía Kafka usando un modelo serializado (`.pkl`), actualizando alertas de riesgo alto en vivo en Streamlit.

## 3. Pipeline, Calidad y Modelo ML
* **DAG (`dags/etl_suicidios_dag.py`):** `etl_suicidios_valle_v2` (`@daily`). Los registros que fallan Great Expectations van a `public.cuarentena_suicidios`.
* **Modelo (`models/train_model.py`):** `RandomForestClassifier` en `Pipeline` de Scikit-Learn (`models/modelo_riesgo_suicidio.pkl`) con división temporal (`<= 2022` train, `>= 2023` test).
* **Métricas (Test >= 2023):** ROC-AUC `0.7401` | Recall Riesgo Alto `0.87` | F1-Score `0.46` | Accuracy `0.5844`.
* **Limitaciones:** Variables de scraping cubren solo el Valle del Cauca; el resto del país se imputa con la mediana.

## 4. Ejecución
    docker-compose up -d --build
    source venv/bin/activate
    python models/train_model.py
    python reports/generate_static_report.py
    python -u streaming/kafka_consumer.py &
    streamlit run streaming/dashboard_realtime.py --server.port 8501
    python scripts/kafka_producer.py
