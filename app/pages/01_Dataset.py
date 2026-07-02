import streamlit as st
import pandas as pd
import json
import os

st.title("📂 Dataset")

@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    base = os.path.normpath(base)
    with open(os.path.join(base, 'data/raw/streaming_users_dirty.json'), 'r', encoding='utf-8') as f:
        raw = pd.DataFrame(json.load(f))
    clean = pd.read_csv(os.path.join(base, 'data/processed/streaming_users_clean.csv'),
                        parse_dates=['last_login_date'])
    return raw, clean

raw, clean = load_data()

st.subheader("Descripción general")
st.write("""
El dataset `streaming_users_dirty.json` contiene registros de usuarios de una plataforma de streaming
latinoamericana. Cada registro representa un usuario con su perfil demográfico y de comportamiento.
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Variables", str(raw.shape[1]))
col2.metric("Registros originales", f"{raw.shape[0]:,}")
col3.metric("Registros limpios", f"{clean.shape[0]:,}")
col4.metric("Retención", f"{clean.shape[0]/raw.shape[0]*100:.1f}%")

st.markdown("---")
st.subheader("Descripción de variables")
vars_df = pd.DataFrame({
    "Variable": ["user_id", "age", "subscription_plan", "monthly_watch_time_mins",
                 "country", "favorite_genre", "last_login_date", "customer_support_tickets"],
    "Tipo": ["int", "int", "str", "float", "str", "str", "date", "int"],
    "Descripción": [
        "Identificador único del usuario",
        "Edad del usuario en años",
        "Plan contratado: Básico, Estándar o Premium",
        "Minutos totales de visualización en el mes",
        "País de residencia del usuario",
        "Género de contenido favorito",
        "Fecha del último inicio de sesión",
        "Tickets de soporte técnico generados"
    ]
})
st.table(vars_df)

st.markdown("---")
st.subheader("Resumen de calidad del dataset original")
col1, col2, col3 = st.columns(3)
col1.metric("Valores nulos", "753")
col2.metric("Duplicados eliminados", "160")
col3.metric("Variantes de texto", "55+")

st.write("""
**Principales problemas detectados:**
- Valores negativos e imposibles en `age`, `monthly_watch_time_mins` y `customer_support_tickets`
- Múltiples variantes de texto para el mismo valor en `subscription_plan`, `country` y `favorite_genre`
- Fechas futuras en `last_login_date` (posteriores a la fecha de corte del dataset)
- Filas duplicadas exactas y por `user_id`
""")

st.markdown("---")
st.subheader("Vista previa del dataset procesado")
st.dataframe(clean.head(50), use_container_width=True)

st.markdown("---")
st.subheader("Log de transformaciones")
try:
    base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    log = pd.read_csv(os.path.join(base, 'logs/pipeline_log.csv'))
    st.dataframe(log, use_container_width=True)
except FileNotFoundError:
    st.info("Ejecutar el notebook 02_calidad_y_limpieza.ipynb para generar el log ETL.")