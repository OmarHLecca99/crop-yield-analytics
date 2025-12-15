# =====================================================
# IMPORTS
# =====================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(
    page_title="Crop Yield Analytics",
    layout="wide"
)

# =====================================================
# RUTAS
# =====================================================
RAW_DATA_PATH = Path("data/raw")
RESULTADOS_PATH = Path("data/resultados")
CONFIG_PATH = Path("data/config")

RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
RESULTADOS_PATH.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.mkdir(parents=True, exist_ok=True)

EXCEL_TEST = RESULTADOS_PATH / "Resultados_Test_Modelo.xlsx"
PRECIOS_CSV = CONFIG_PATH / "precios_cultivos.csv"


# =====================================================
# SESSION STATE
# =====================================================
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None


# =====================================================
# UTILIDADES
# =====================================================
def get_latest_csv(folder: Path):
    files = list(folder.glob("*.csv"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def cargar_precios():
    if not PRECIOS_CSV.exists():
        df_default = pd.DataFrame({
            "Crop": ["Rice", "Wheat", "Cotton", "Soybean", "Barley", "Maize"],
            "Precio_USD": [450, 280, 1800, 520, 240, 260]
        })
        df_default.to_csv(PRECIOS_CSV, index=False)
    return pd.read_csv(PRECIOS_CSV)


# =====================================================
# TABS
# =====================================================
tab_upload, tab_eda, tab_experiments, tab_model = st.tabs([
    "📤 Carga de Datos",
    "🔍 Exploración",
    "🧪 Experimentos",
    "🥇 Mejor Modelo"
])


# =====================================================
# TAB 1 – CARGA DE DATOS
# =====================================================
with tab_upload:
    st.header("📤 Carga de Dataset")

    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_uploaded_filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = RAW_DATA_PATH / f"{ts}_{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.success(f"Archivo cargado: {save_path.name}")
        else:
            st.info("El archivo ya fue cargado previamente.")

    if st.button("👁️ Ver vista previa"):
        latest = get_latest_csv(RAW_DATA_PATH)
        if latest:
            df = pd.read_csv(latest)
            st.metric("Total de registros", len(df))
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.warning("No hay archivos cargados.")


# =====================================================
# TAB 2 – EDA
# =====================================================
with tab_eda:
    st.header("🔍 Exploración de Datos")

    latest = get_latest_csv(RAW_DATA_PATH)
    if not latest:
        st.warning("Primero cargue un dataset.")
    else:
        df = pd.read_csv(latest)
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

        st.subheader("📊 Estadísticas descriptivas")
        st.dataframe(df[num_cols].describe(), use_container_width=True)

        st.subheader("📈 Histograma")
        col_hist = st.selectbox("Variable", num_cols)
        bins = st.slider("Bins", 5, 100, 20)
        fig = px.histogram(df, x=col_hist, nbins=bins)
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Dispersión")
        col_x = st.selectbox("X", num_cols)
        col_y = st.selectbox("Y", num_cols, index=1 if len(num_cols) > 1 else 0)
        fig = px.scatter(df, x=col_x, y=col_y, opacity=0.4)
        fig.update_traces(marker=dict(symbol="square", size=6))
        st.plotly_chart(fig, use_container_width=True)

        if cat_cols:
            st.subheader("📊 Frecuencia categórica")
            col_cat = st.selectbox("Categórica", cat_cols)
            freq = df[col_cat].value_counts().reset_index()
            fig = px.bar(freq, x="index", y=col_cat)
            st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 3 – EXPERIMENTOS (MongoDB)
# =====================================================
with tab_experiments:
    st.header("🧪 Experimentos de Modelos")

    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = "crop_yield"
    COLLECTION_NAME = "experimentos"

    if not MONGO_URI:
        st.error("MONGO_URI no configurado en Secrets")
        st.stop()

    csv_path = RESULTADOS_PATH / "experimentos_modelos.csv"

    if st.button("📥 Descargar desde MongoDB"):
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            col = client[DB_NAME][COLLECTION_NAME]
            df_exp = pd.DataFrame(list(col.find({}, {"_id": 0})))
            client.close()

            if df_exp.empty:
                st.warning("No hay datos en MongoDB.")
            else:
                df_exp = df_exp.rename(columns={
                    "r2_mean": "r2",
                    "mae_mean": "mae",
                    "rmse_mean": "rmse"
                }).drop(columns=["nombre_run", "integrante", "fuente"], errors="ignore")

                df_exp.to_csv(csv_path, index=False)
                st.success("Experimentos descargados.")
        except Exception as e:
            st.error(str(e))

    if csv_path.exists():
        df_exp = pd.read_csv(csv_path)
        st.metric("Total Experimentos", len(df_exp))

        df_exp["modelo_index"] = df_exp["modelo"] + "_" + df_exp.index.astype(str)

        st.subheader("📋 Tabla de Experimentos")
        st.dataframe(df_exp.head(20), use_container_width=True)

        st.subheader("📈 Métricas")
        for metric in ["r2", "mae", "rmse"]:
            fig = px.bar(
                df_exp.sort_values(metric),
                x=metric,
                y="modelo_index",
                orientation="h",
                title=metric.upper()
            )
            st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 4 – MEJOR MODELO (BUSINESS VALUE)
# =====================================================
with tab_model:
    st.header("🥇 Mejor Modelo – Business Value")

    df_precios = cargar_precios()
    edited = st.data_editor(df_precios, use_container_width=True)

    if st.button("💾 Guardar precios"):
        edited.to_csv(PRECIOS_CSV, index=False)
        st.success("Precios actualizados.")

    if not EXCEL_TEST.exists():
        st.warning("No se encontró Resultados_Test_Modelo.xlsx")
    else:
        df = pd.read_excel(EXCEL_TEST)
        df = df.merge(
            edited.rename(columns={"Precio_USD": "Precio_ton"}),
            on="Crop",
            how="left"
        )

        df["F_error"] = np.where(df["Predicho"] > df["Valor_Real"], 1.0, 0.5)
        df["Termino_ingreso"] = df["Precio_ton"] * df["Predicho"]
        df["Costo_error"] = df["Precio_ton"] * df["F_error"]
        df["Termino_penalidad"] = df["Costo_error"] * abs(df["Valor_Real"] - df["Predicho"])
        df["Business_Value"] = df["Termino_ingreso"] - df["Termino_penalidad"]

        BV_total = df["Business_Value"].sum()
        BV_ideal = (df["Precio_ton"] * df["Valor_Real"]).sum()

        BV_mm = BV_total / 1e6
        BV_ideal_mm = BV_ideal / 1e6

        df["BV_rel"] = ((df["Predicho"] - df["Valor_Real"]) / df["Valor_Real"]) * 100
        BV_rel = df["BV_rel"].mean()

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=BV_mm,
                number={"suffix": " MM"},
                title={"text": "Business Value Total"},
                gauge={"axis": {"range": [0, BV_ideal_mm]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=BV_rel,
                number={"suffix": "%"},
                title={"text": "BV Relativo Promedio"},
                gauge={"axis": {"range": [-30, 30]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Vista parcial del set de prueba")
        st.dataframe(df.head(20), use_container_width=True)