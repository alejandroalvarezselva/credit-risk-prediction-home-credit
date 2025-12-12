import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# Rutas a los artefactos
# =========================
BASE_DIR = Path(__file__).resolve().parent

PIPELINE_PATH = BASE_DIR / "artifacts" / "preprocessing_pipeline_v3.joblib"
FEATURES_PATH = BASE_DIR / "artifacts" / "feature_names_v3.npy"
MODEL_PATH = BASE_DIR / "artifacts" / "lgbm_optimized_v1.joblib"


# =========================
# Carga de artefactos (cacheada)
# =========================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(PIPELINE_PATH)
    feature_names = np.load(FEATURES_PATH, allow_pickle=True)
    model = joblib.load(MODEL_PATH)
    return pipeline, feature_names, model


# =========================
# Función de predicción
# =========================
def score_clients(df_raw: pd.DataFrame,
                  pipeline,
                  model,
                  id_column: str = "SK_ID_CURR") -> pd.DataFrame:
    """
    Aplica el pipeline y el modelo al DataFrame df_raw y devuelve
    un DataFrame con las probabilidades de impago.
    """

    # Aplicar el pipeline (puede devolver matriz densa o sparse)
    X_pre = pipeline.transform(df_raw)

    # Predecir probabilidad de TARGET = 1 (impago)
    probs = model.predict_proba(X_pre)[:, 1]

    # Construir resultado
    result = pd.DataFrame()

    if id_column in df_raw.columns:
        result[id_column] = df_raw[id_column]
    else:
        # Si no hay columna de ID, usamos el índice como identificador
        result["row_id"] = df_raw.index

    result["TARGET"] = probs  # probabilidad de impago

    return result


# =========================
# Interfaz Streamlit
# =========================
def main():
    st.set_page_config(
        page_title="Credit Risk Prediction – Home Credit",
        layout="wide",
    )

    st.title("Credit Risk Prediction – Home Credit")
    st.write(
        "Sube un fichero CSV con los datos de los clientes (mismo esquema que el "
        "dataset enriquecido de test) y la aplicación devolverá la **probabilidad "
        "de impago (TARGET)** para cada fila."
    )

    # Cargar artefactos
    try:
        pipeline, feature_names, model = load_artifacts()
        st.success("✅ Artefactos cargados correctamente.")
    except Exception as e:
        st.error("❌ Error cargando los artefactos. Revisa la carpeta 'artifacts/'.")
        st.exception(e)
        return

    st.divider()

    # Subida de fichero
    uploaded_file = st.file_uploader(
        "📁 Sube un archivo CSV con los datos de los clientes",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error("❌ No se ha podido leer el CSV. Comprueba el formato.")
            st.exception(e)
            return

        st.subheader("Vista previa de los datos cargados")
        st.dataframe(df_raw.head())

        # Botón para lanzar el scoring
        if st.button("Calcular probabilidad de impago"):
            with st.spinner("Calculando scores..."):
                try:
                    df_scores = score_clients(df_raw, pipeline, model)
                except Exception as e:
                    st.error(
                        "❌ Error al aplicar el pipeline o el modelo. "
                        "Comprueba que las columnas del CSV coinciden con las del dataset enriquecido."
                    )
                    st.exception(e)
                    return

            st.subheader("Resultados de scoring")
            st.write("Probabilidad estimada de impago (TARGET) para cada registro:")
            st.dataframe(df_scores.head())

            # Descargar resultados
            csv_bytes = df_scores.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Descargar resultados en CSV",
                data=csv_bytes,
                file_name="predicciones_home_credit.csv",
                mime="text/csv",
            )
    else:
        st.info("⬆️ Sube un CSV para empezar.")


if __name__ == "__main__":
    main()
