import streamlit as st

st.set_page_config(page_title="Streaming Users - Minería de Datos", page_icon="📺", layout="wide")

st.title("📺 Análisis de Usuarios de Plataforma de Streaming")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Contexto del proyecto")
    st.write("""
    Este proyecto analiza un dataset de usuarios de una plataforma de streaming latinoamericana.
    El dataset contiene registros de usuarios de 7 países, con información sobre su plan de suscripción,
    tiempo de visualización mensual, género favorito, fecha de último acceso y cantidad de tickets
    de soporte generados.

    El objetivo es explorar la estructura del dataset, realizar un proceso de limpieza documentado,
    analizar patrones de consumo y comportamiento, y reducir dimensionalidad mediante PCA.
    """)

    st.subheader("Integrantes")
    st.info("""
    **Grupo N°:** [Completar]
    
    - Integrante 1: [Nombre y Apellido]
    - Integrante 2: [Nombre y Apellido]
    
    **Comisión:** [Completar] | **Fecha de entrega:** [Completar]
    """)

with col2:
    st.subheader("Resumen del dataset")
    st.metric("Registros originales", "8.160")
    st.metric("Variables", "8")
    st.metric("Países", "7")
    st.metric("Planes de suscripción", "3")

st.markdown("---")
st.markdown("🔗 [Repositorio GitHub](https://github.com/[usuario]/PI_Mineria_Datos_1) &nbsp;|&nbsp; 📓 Notebooks disponibles en `/notebooks/`")
