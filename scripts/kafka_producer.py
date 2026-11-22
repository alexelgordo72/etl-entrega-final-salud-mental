import os
import json
import time
import uuid
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine
from kafka import KafkaProducer

def publish_events_to_kafka(limit=100, delay=0.3, **kwargs):
    user = os.getenv("DW_USER", "airflow")
    password = os.getenv("DW_PASSWORD", "airflow")
    host = os.getenv("DW_HOST", "host.docker.internal")
    port = os.getenv("DW_PORT", "5433")
    db = os.getenv("DW_DB", "suicidios_colombia")
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    topic = "suicidios-riesgo-stream"

    # 1. Leer tabla de hechos desde el Data Warehouse
    try:
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")
        df = pd.read_sql("SELECT * FROM public.fact_suicidios_enriquecida_v2", engine)
    except Exception as e:
        print(f"Aviso leyendo DW ({e}), usando respaldo CSV...")
        df = pd.read_csv("/opt/airflow/data/dataset_limpio.csv")

    cols_base = [c for c in df.columns if not c.endswith("_x") and not c.endswith("_y")]
    df = df[cols_base].copy()

    # 2. Preparar registros al grano del modelo (SIN variable objetivo)
    group_cols = ["anio", "mes", "departamento", "municipio", "zona_hecho"]
    events_df = (
        df.groupby(group_cols, dropna=False)
        .agg(
            tasa_desempleo=("tasa_desempleo", "mean"),
            total_festivos_anio=("total_festivos_anio", "first"),
            superficie_km2=("superficie_km2", "first"),
            poblacion_2018=("poblacion_2018", "first"),
            altitud_msnm=("altitud_msnm", "first"),
        )
        .reset_index()
    )

    # Priorizar registros recientes (2023-2024) para simular monitoreo actual
    recientes = events_df[events_df["anio"] >= 2023]
    muestra = recientes.sample(n=min(limit, len(recientes)), random_state=42) if not recientes.empty else events_df.head(limit)

    # 3. Conectar al broker de Kafka y publicar eventos JSON
    producer = KafkaProducer(
        bootstrap_servers=[bootstrap],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    enviados = 0
    for _, row in muestra.iterrows():
        evento = {
            "event_id": str(uuid.uuid4())[:8],
            "event_timestamp": datetime.now(timezone.utc).isoformat(),
            "anio": int(row["anio"]) if pd.notna(row["anio"]) else None,
            "mes": str(row["mes"]) if pd.notna(row["mes"]) else None,
            "departamento": str(row["departamento"]) if pd.notna(row["departamento"]) else None,
            "municipio": str(row["municipio"]) if pd.notna(row["municipio"]) else None,
            "zona_hecho": str(row["zona_hecho"]) if pd.notna(row["zona_hecho"]) else None,
            "tasa_desempleo": float(row["tasa_desempleo"]) if pd.notna(row["tasa_desempleo"]) else None,
            "total_festivos_anio": float(row["total_festivos_anio"]) if pd.notna(row["total_festivos_anio"]) else None,
            "superficie_km2": float(row["superficie_km2"]) if pd.notna(row["superficie_km2"]) else None,
            "poblacion_2018": float(row["poblacion_2018"]) if pd.notna(row["poblacion_2018"]) else None,
            "altitud_msnm": float(row["altitud_msnm"]) if pd.notna(row["altitud_msnm"]) else None,
        }
        producer.send(topic, value=evento)
        enviados += 1
        time.sleep(delay)

    producer.flush()
    producer.close()
    print(f"ÉXITO: {enviados} eventos publicados en el tópico '{topic}'.")

if __name__ == "__main__":
    os.environ["DW_HOST"] = "localhost"
    os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9092"
    publish_events_to_kafka(limit=30, delay=0.2)
