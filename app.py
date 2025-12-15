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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymongo import MongoClient

# =====================================================
# CONFIGURACIÓN INICIAL
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
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=lambda f: f.stat().st_mtime)

def cargar_precios():
    if not PRECIOS_CSV.exists():
        df_default = pd.DataFrame({
            "Crop": ["Rice", "Wheat", "Cotton", "Soybean", "Barley", "Maize"],
            "Precio_USD": [450, 280, 1800, 520, 240, 260]
        })
        df_default.to_csv(PRECIOS_CSV, index=False)
    return pd.read_csv(PRECIOS_CSV)


st.title("🌾 Crop Yield Analytics Dashboard")

st.markdown(
    """
    **Plataforma analítica para evaluación de modelos predictivos de rendimiento agrícola**  
    *Exploración de datos, comparación experimental y evaluación de Business Value*
    """
)

st.divider()

# =====================================================
# TABS PRINCIPALES
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

    uploaded_file = st.file_uploader(
        "Sube un archivo CSV",
        type=["csv"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_uploaded_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = RAW_DATA_PATH / f"{timestamp}_{uploaded_file.name}"

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.last_uploaded_filename = uploaded_file.name
            st.success(f"✅ Archivo cargado correctamente en: data/raw/{save_path.name}")
        else:
            st.info("ℹ️ El archivo ya fue cargado anteriormente.")

    st.divider()

    if st.button("👁️ Ver vista previa del último archivo cargado"):
        latest_csv = get_latest_csv(RAW_DATA_PATH)

        if latest_csv is None:
            st.warning("⚠️ No hay archivos CSV en data/raw/")
        else:
            st.info(f"Mostrando vista previa de: {latest_csv.name}")
            try:
                df_preview = pd.read_csv(latest_csv)

                # ============================
                # KPI – Total de registros
                # ============================
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(
                        label="📊 Total de registros",
                        value=f"{len(df_preview):,}"
                    )

                # ============================
                # Vista previa
                # ============================
                st.dataframe(
                    df_preview.head(10),
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {e}")

# =====================================================
# TAB 2 – EXPLORACIÓN DE DATOS (EDA)
# =====================================================
with tab_eda:
    st.header("🔍 Exploración de Datos (EDA)")

    latest_csv = get_latest_csv(RAW_DATA_PATH)

    if latest_csv is None:
        st.warning("⚠️ Primero cargue un dataset en la pestaña 'Carga de Datos'.")
    else:
        df = pd.read_csv(latest_csv)

        # ============================
        # Separar tipos de columnas
        # ============================
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

        # ============================
        # 1. Estadísticas descriptivas
        # ============================
        st.subheader("📊 Estadísticas descriptivas (variables numéricas)")

        selected_stats_cols = st.multiselect(
            "Selecciona columnas numéricas",
            options=num_cols,
            default=num_cols
        )

        if selected_stats_cols:
            st.dataframe(
                df[selected_stats_cols].describe(),
                use_container_width=True
            )
        else:
            st.info("Selecciona al menos una columna numérica.")

        st.divider()

        # ============================
        # 2. Histograma
        # ============================
        st.subheader("📈 Histograma")

        col_hist = st.selectbox(
            "Selecciona variable numérica",
            options=num_cols,
            key="hist_col"
        )

        bins = st.slider(
            "Número de bins",
            min_value=5,
            max_value=100,
            value=20
        )

        fig_hist = px.histogram(
            df,
            x=col_hist,
            nbins=bins,
            title=f"Distribución de {col_hist}"
        )

        fig_hist.update_traces(
            marker_line_width=1.5,
            marker_line_color="black"
        )        

        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # ============================
        # 3. Gráfico de dispersión
        # ============================
        st.subheader("🎯 Gráfico de dispersión")

        col_x = st.selectbox(
            "Variable X",
            options=num_cols,
            key="scatter_x"
        )

        col_y = st.selectbox(
            "Variable Y",
            options=num_cols,
            index=1 if len(num_cols) > 1 else 0,
            key="scatter_y"
        )

        fig_scatter = px.scatter(
            df,
            x=col_x,
            y=col_y,
            title=f"{col_x} vs {col_y}",
            opacity=0.4
        )

        fig_scatter.update_traces(
            marker=dict(
                symbol="square",
                size=6,
                line=dict(width=0.5, color="black")
            )
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # ============================
        # 4. Gráfico de barras (categóricas)
        # ============================
        st.subheader("📊 Frecuencia de variables categóricas")

        if cat_cols:
            col_cat = st.selectbox(
                "Selecciona variable categórica",
                options=cat_cols
            )

            freq_df = df[col_cat].value_counts().reset_index()
            freq_df.columns = [col_cat, "Frecuencia"]

            fig_bar = px.bar(
                freq_df,
                x=col_cat,
                y="Frecuencia",
                title=f"Frecuencia de {col_cat}"
            )
            
            # ============================
            # Ajuste de eje Y (zoom)
            # ============================
            y_min = freq_df["Frecuencia"].min()
            y_max = freq_df["Frecuencia"].max()

            # Margen controlado (ej. ±5%)
            margen = (y_max - y_min) * 0.1

            fig_bar.update_yaxes(
                range=[
                    y_min - margen,
                    y_max + margen
                ]
            )

            fig_bar.update_traces(
                marker_line_width=1,
                marker_line_color="black"
            )

            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No se detectaron columnas categóricas en el dataset.")

# =====================================================
# TAB 3 – EXPERIMENTOS (MongoDB)
# =====================================================
with tab_experiments:
    st.header("🥇 Experimentos de Modelos de Regresión")

    st.markdown(
        """
        En esta sección se consolidan los resultados de **todos los experimentos**
        ejecutados durante la etapa de modelado, almacenados en MongoDB.
        """
    )

    # ============================
    # CONFIG MONGODB
    # ============================

    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = "crop_yield"
    COLLECTION_NAME = "experimentos"

    csv_path = RESULTADOS_PATH / "experimentos_modelos.csv"

    # ============================
    # BOTÓN 1 – DESCARGAR EXPERIMENTOS
    # ============================
    if st.button("📥 Descargar experimentos desde MongoDB"):
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[DB_NAME]
            col = db[COLLECTION_NAME]

            data = list(col.find({}, {"_id": 0}))
            df_exp = pd.DataFrame(data)

            client.close()

            if df_exp.empty:
                st.warning("⚠️ No se encontraron experimentos en MongoDB.")
            else:
                # Normalización (heredada del código original)
                df_exp = df_exp.rename(columns={
                    "r2_mean": "r2",
                    "mae_mean": "mae",
                    "rmse_mean": "rmse"
                })

                # Eliminación de columnas NO deseadas
                df_exp = df_exp.drop(
                    columns=["nombre_run", "integrante", "fuente"],
                    errors="ignore"
                )

                df_exp.to_csv(csv_path, index=False)

                st.success(f"✅ Experimentos guardados en: {csv_path}")

                st.download_button(
                    label="⬇️ Descargar CSV de experimentos",
                    data=df_exp.to_csv(index=False),
                    file_name="experimentos_modelos.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error al conectar con MongoDB: {e}")

    st.divider()

    # ============================
    # BOTÓN 2 – VISUALIZAR RESULTADOS
    # ============================
    if st.button("📊 Visualizar comparativa de métricas"):
        if not csv_path.exists():
            st.warning("⚠️ Primero descargue los experimentos desde MongoDB.")
        else:
            df_exp = pd.read_csv(csv_path)

            # Contar el número total de registros
            total_experimentos = len(df_exp)

            # Mostrar el indicador con la cantidad de experimentos
            st.metric("Total de Experimentos", total_experimentos)

            # Conversión segura a numérico
            for col in ["r2", "mae", "rmse"]:
                df_exp[col] = pd.to_numeric(df_exp[col], errors="coerce")

            df_exp = df_exp.dropna(subset=["r2", "mae", "rmse"])

            # -----------------------------------------------------
            # Crear identificador legible por experimento
            # -----------------------------------------------------
            df_exp = df_exp.reset_index(drop=True)
            df_exp["modelo_index"] = df_exp["modelo"] + "_" + df_exp.index.astype(str)

            # =====================================================
            # TABLA DE EXPERIMENTOS (ANTES DE LOS GRÁFICOS)
            # =====================================================
            st.subheader("📋 Tabla consolidada de experimentos")

            # Resaltado de columna R2
            styled_df = df_exp.style.background_gradient(
                subset=["r2"],
                cmap="YlGn"
            ).format({
                "r2": "{:.6f}",
                "mae": "{:.6f}",
                "rmse": "{:.6f}"
            })

            st.dataframe(
                styled_df,
                use_container_width=True,
                height=420
            )

            st.divider()

            # =====================================================
            # GRÁFICOS
            # =====================================================
            st.subheader("📈 Comparativa global de métricas")

            # Rangos dinámicos para mejorar visualización
            r2_min, r2_max = df_exp["r2"].min(), df_exp["r2"].max()
            mae_min, mae_max = df_exp["mae"].min(), df_exp["mae"].max()
            rmse_min, rmse_max = df_exp["rmse"].min(), df_exp["rmse"].max()

            # Margen visual (5%)
            def expand_range(min_v, max_v, pct=0.05):
                delta = (max_v - min_v) * pct
                return [min_v - delta, max_v + delta]

            # ============================
            # R² (mayor es mejor)
            # ============================
            df_r2_top = (
                df_exp
                .sort_values("r2", ascending=False)
                .head(15)
            )

            fig_r2 = px.bar(
                df_r2_top.sort_values("r2", ascending=True),
                x="r2",
                y="modelo_index",
                orientation="h",
                title="R² – TOP 15 experimentos"
            )

            fig_r2.update_xaxes(
                range=expand_range(r2_min, r2_max),
                tickformat=".4f"
            )
            
            st.plotly_chart(fig_r2, use_container_width=True)

            # ============================
            # MAE (menor es mejor)
            # ============================
            df_mae_top = (
                df_exp
                .sort_values("mae", ascending=True)
                .head(15)
            )

            fig_mae = px.bar(
                df_mae_top.sort_values("mae", ascending=False),
                x="mae",
                y="modelo_index",
                orientation="h",
                title="MAE – TOP 15 experimentos"
            )

            fig_mae.update_xaxes(
                range=expand_range(mae_min, mae_max),
                tickformat=".4f"
            )

            st.plotly_chart(fig_mae, use_container_width=True)

            # ============================
            # RMSE (menor es mejor)
            # ============================
            df_rmse_top = (
                df_exp
                .sort_values("rmse", ascending=True)
                .head(15)
            )

            fig_rmse = px.bar(
                df_rmse_top.sort_values("rmse", ascending=False),
                x="rmse",
                y="modelo_index",
                orientation="h",
                title="RMSE – TOP 15 experimentos"
            )

            fig_rmse.update_xaxes(
                range=expand_range(rmse_min, rmse_max),
                tickformat=".4f"
            )

            st.plotly_chart(fig_rmse, use_container_width=True)


# =====================================================
# TAB 4 – MEJOR MODELO (BUSINESS VALUE)
# =====================================================
with tab_model:
    st.header("🥇 Mejor Modelo – Business Value")

    st.markdown(
        """
        Esta sección evalúa el **impacto económico del mejor modelo (Ridge)**.
        """
    )

    # =================================================
    # PRECIOS POR CULTIVO (CONFIGURABLES)
    # =================================================
    st.subheader("💼 Parámetros de negocio – Precios por cultivo")

    df_precios = cargar_precios()

    edited_precios = st.data_editor(
        df_precios,
        use_container_width=True,
        num_rows="fixed"
    )

    if st.button("💾 Guardar precios"):
        edited_precios["Precio_USD"] = pd.to_numeric(
            edited_precios["Precio_USD"], errors="coerce"
        )
        edited_precios.to_csv(PRECIOS_CSV, index=False)
        st.success("✅ Precios actualizados correctamente")

    st.divider()

    # =================================================
    # LECTURA RESULTADOS DEL MODELO
    # =================================================
    if not EXCEL_TEST.exists():
        st.warning("⚠️ No se encontró el archivo Resultados_Test_Modelo.xlsx")
    else:
        df = pd.read_excel(EXCEL_TEST)

        # Merge con precios
        df = df.merge(
            edited_precios.rename(columns={"Precio_USD": "Precio_ton"}),
            on="Crop",
            how="left"
        )

        # =================================================
        # CÁLCULO BUSINESS VALUE (MISMA FÓRMULA DEL CODE BASE)
        # =================================================
        df["F_error"] = np.where(
            df["Predicho"] > df["Valor_Real"],
            1.0,   # sobreestimación
            0.5    # subestimación
        )

        df["Termino_ingreso"] = df["Precio_ton"] * df["Predicho"]
        df["Costo_error"] = df["Precio_ton"] * df["F_error"]
        df["Termino_penalidad"] = df["Costo_error"] * np.abs(
            df["Valor_Real"] - df["Predicho"]
        )

        df["Business_Value"] = df["Termino_ingreso"] - df["Termino_penalidad"]

        # =================================================
        # AGREGADOS (NO PROMEDIOS)
        # =================================================
        BV_total = df["Business_Value"].sum()

        df["BV_ideal"] = df["Precio_ton"] * df["Valor_Real"]
        BV_ideal_total = df["BV_ideal"].sum()

        df["BV_rel"] = ((df["Predicho"] - df["Valor_Real"]) / df["Valor_Real"]) * 100
        BV_relativo = df["BV_rel"].mean()

        # Conversión a millones
        BV_total_mm = BV_total / 1e6
        BV_ideal_mm = BV_ideal_total / 1e6

        # =================================================
        # INDICADORES
        # =================================================
        st.subheader("📊 Indicadores de Business Value")

        col1, col2 = st.columns(2)

        # -------- BV USD --------
        with col1:
            fig_bv_usd = go.Figure(go.Indicator(
                mode="gauge+number",
                value=BV_total_mm,
                title={"text": "Business Value Total (USD)"},
                number={
                    "suffix": " MM",
                    "valueformat": ".3f"
                },
                gauge={
                    "axis": {"range": [0, BV_ideal_mm]},
                    "bar": {"color": "black"},
                    "steps": [
                        {"range": [0, BV_ideal_mm * 0.8], "color": "red"},
                        {"range": [BV_ideal_mm * 0.8, BV_ideal_mm * 0.9], "color": "yellow"},
                        {"range": [BV_ideal_mm * 0.9, BV_ideal_mm], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "value": BV_total_mm
                    }
                }
            ))
            st.plotly_chart(fig_bv_usd, use_container_width=True)

        # -------- BV RELATIVO --------
        with col2:
            fig_bv_rel = go.Figure(go.Indicator(
                mode="gauge+number",
                value=BV_relativo,
                title={"text": "BV Relativo Promedio (%)"},
                number={"suffix": "%", "valueformat": ".2f"},
                gauge={
                    "axis": {"range": [-30, 30]},
                    "bar": {"color": "black"},
                    "steps": [
                        {"range": [-30, -15], "color": "red"},
                        {"range": [-15, -10], "color": "yellow"},
                        {"range": [-10, 5], "color": "lightgreen"},
                        {"range": [5, 10], "color": "yellow"},
                        {"range": [10, 30], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "value": BV_relativo
                    }
                }
            ))
            st.plotly_chart(fig_bv_rel, use_container_width=True)

        st.divider()

        # =================================================
        # TABLA (LIMITADA)
        # =================================================
        st.subheader("📋 Vista parcial del set de prueba (head 20)")

        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=420
        )

        st.divider()

        # =================================================
        # INTERPRETACIÓN
        # =================================================
        st.markdown(
            f"""
            **Interpretación de negocio**

            - El **Business Value total** generado por el modelo es de  
              **USD {BV_total:,.2f}** (≈ {BV_total_mm:.3f} MM).
            - El **BV relativo promedio** es de **{BV_relativo:.2f}%**, indicando
              una desviación controlada respecto al escenario ideal.
            - La externalización de precios permite evaluar distintos escenarios
              económicos sin modificar el modelo predictivo.
            """
        )
