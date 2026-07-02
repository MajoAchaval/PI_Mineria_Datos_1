import streamlit as st
import pandas as pd

st.title("📝 Conclusiones")

st.subheader("Hallazgos principales")
st.write("""
A partir del análisis exploratorio del dataset de usuarios de la plataforma de streaming, se obtuvieron los siguientes hallazgos:

**1. Perfil demográfico:** [Completar con valores reales — rango etario dominante, distribución por país]

**2. Consumo por plan:** [Completar — ¿El plan Premium tiene mayor watch_time? ¿En qué magnitud?]

**3. Soporte vs. consumo:** [Completar — ¿Existe correlación negativa entre tickets y watch_time?]

**4. Distribución geográfica:** [Completar — ¿Qué país concentra más usuarios?]

**5. PCA:** [Completar — ¿Cuántos componentes explican el 80%? ¿Qué variable domina cada componente?]
""")

st.markdown("---")
st.subheader("Limitaciones")
st.write("""
- El dataset no incluye información sobre el contenido consumido ni la fecha de alta del usuario,
  lo que limita el análisis de comportamiento temporal.
- Las fechas futuras y valores imposibles eliminados durante la limpieza podrían representar
  un sesgo sistemático no identificable con la información disponible.
- El alcance de las conclusiones está condicionado por la información disponible y las decisiones
  documentadas durante el proceso ETL.
""")

st.markdown("---")
st.subheader("Próximos pasos")
st.write("""
- Incorporar variables de contenido (títulos vistos, minutos por género) para profundizar el
  análisis de preferencias.
- Aplicar clustering (K-Means) sobre los componentes del PCA para segmentar perfiles de usuario.
- Integrar datos longitudinales para analizar la evolución del comportamiento a lo largo del tiempo.
""")

st.markdown("---")
col1, col2 = st.columns(2)
col1.markdown("🔗 [Repositorio GitHub](https://github.com/[usuario]/PI_Mineria_Datos_1)")
col2.markdown("🌐 [Aplicación en Streamlit Cloud](https://[app].streamlit.app)")
