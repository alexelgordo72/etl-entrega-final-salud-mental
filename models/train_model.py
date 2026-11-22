import os
import json
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score

# 1. Cargar datos desde el Data Warehouse (con respaldo en CSV)
def cargar_datos():
    try:
        engine = create_engine("postgresql+psycopg2://airflow:airflow@localhost:5433/suicidios_colombia")
        df = pd.read_sql("SELECT * FROM public.fact_suicidios_enriquecida_v2", engine)
        print(f"Datos cargados desde PostgreSQL: {len(df)} filas.")
    except Exception as e:
        print(f"Aviso: No se pudo leer PostgreSQL ({e}). Leyendo CSV local...")
        df = pd.read_csv("data/dataset_limpio.csv")
        print(f"Datos cargados desde CSV: {len(df)} filas.")

    # Limpiar columnas duplicadas (_x, _y) si existen
    cols_base = [c for c in df.columns if not c.endswith("_x") and not c.endswith("_y")]
    df = df[cols_base].copy()
    return df

def main():
    df = cargar_datos()

    # 2. Agregar al grano Municipio-Año-Mes-Zona para definir el target sin data leakage
    group_cols = ["anio", "mes", "departamento", "municipio", "zona_hecho"]
    agg_df = (
        df.groupby(group_cols, dropna=False)
        .agg(
            total_casos=("cantidad", "sum"),
            tasa_desempleo=("tasa_desempleo", "mean"),
            total_festivos_anio=("total_festivos_anio", "first"),
            superficie_km2=("superficie_km2", "first"),
            poblacion_2018=("poblacion_2018", "first"),
            altitud_msnm=("altitud_msnm", "first"),
        )
        .reset_index()
    )

    # Umbral de riesgo alto: percentil 75 (mínimo 2 casos en el mes/municipio/zona)
    umbral = max(2, int(agg_df["total_casos"].quantile(0.75)))
    agg_df["riesgo_alto"] = (agg_df["total_casos"] >= umbral).astype(int)

    num_features = [
        "tasa_desempleo",
        "total_festivos_anio",
        "superficie_km2",
        "poblacion_2018",
        "altitud_msnm",
    ]
    cat_features = ["departamento", "zona_hecho", "mes"]
    features = num_features + cat_features

    # 3. División temporal (evita fuga de información del futuro)
    train_df = agg_df[agg_df["anio"] <= 2022]
    test_df = agg_df[agg_df["anio"] >= 2023]

    X_train, y_train = train_df[features], train_df["riesgo_alto"]
    X_test, y_test = test_df[features], test_df["riesgo_alto"]

    # 4. Pipeline de preprocesamiento + modelo
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_features),
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # 5. Entrenamiento y evaluación
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "umbral_casos_riesgo_alto": umbral,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "features": features,
    }

    print("\n=== MÉTRICAS EN CONJUNTO DE PRUEBA (AÑOS >= 2023) ===")
    print(classification_report(y_test, y_pred, target_names=["Moderado/Bajo", "Riesgo Alto"]))
    print("Resumen:", json.dumps(metrics, indent=2))

    # 6. Serializar el pipeline completo (.pkl) y las métricas
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/modelo_riesgo_suicidio.pkl")
    with open("models/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModelo serializado en: models/modelo_riesgo_suicidio.pkl")

if __name__ == "__main__":
    main()
