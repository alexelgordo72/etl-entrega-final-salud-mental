import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from kafka import KafkaConsumer

MODEL_PATH = "/home/alexelgordo/etl_entrega_ii/models/modelo_riesgo_suicidio.pkl"
TOPIC = "suicidios-riesgo-stream"
BOOTSTRAP = "localhost:9092"
DB_URI = "postgresql+psycopg2://airflow:airflow@localhost:5433/suicidios_colombia"

FEATURES = [
    "tasa_desempleo",
    "total_festivos_anio",
    "superficie_km2",
    "poblacion_2018",
    "altitud_msnm",
    "departamento",
    "zona_hecho",
    "mes",
]

def init_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.predicciones_tiempo_real (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(50),
                event_timestamp TIMESTAMP,
                anio INT,
                mes VARCHAR(30),
                departamento VARCHAR(100),
                municipio VARCHAR(100),
                zona_hecho VARCHAR(100),
                tasa_desempleo DOUBLE PRECISION,
                prediccion_riesgo INT,
                etiqueta_riesgo VARCHAR(30),
                probabilidad_riesgo DOUBLE PRECISION,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

def main():
    print("Cargando modelo serializado (.pkl)...")
    model = joblib.load(MODEL_PATH)
    engine = create_engine(DB_URI)
    init_table(engine)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BOOTSTRAP],
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="grupo-inferencia-riesgo",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(f"Escuchando eventos en '{TOPIC}'... (Ctrl+C para detener)")
    for message in consumer:
        evento = message.value
        try:
            row_df = pd.DataFrame([{col: evento.get(col) for col in FEATURES}])
            pred = int(model.predict(row_df)[0])
            prob = float(model.predict_proba(row_df)[0][1])
            etiqueta = "Riesgo Alto" if pred == 1 else "Moderado/Bajo"

            registro = {
                "event_id": evento.get("event_id"),
                "event_timestamp": evento.get("event_timestamp"),
                "anio": evento.get("anio"),
                "mes": evento.get("mes"),
                "departamento": evento.get("departamento"),
                "municipio": evento.get("municipio"),
                "zona_hecho": evento.get("zona_hecho"),
                "tasa_desempleo": evento.get("tasa_desempleo"),
                "prediccion_riesgo": pred,
                "etiqueta_riesgo": etiqueta,
                "probabilidad_riesgo": round(prob, 4),
                "processed_at": datetime.now(timezone.utc),
            }

            pd.DataFrame([registro]).to_sql(
                "predicciones_tiempo_real",
                engine,
                schema="public",
                if_exists="append",
                index=False,
            )
            print(f"[{registro['event_id']}] {registro['municipio']} ({registro['departamento']}) -> {etiqueta} (prob={prob:.2%})")
        except Exception as e:
            print(f"Error procesando evento {evento.get('event_id')}: {e}")

if __name__ == "__main__":
    main()
