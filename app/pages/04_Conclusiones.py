import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os

st.title("📝 Conclusiones")

@st.cache_data
def load_data():
    base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    return pd.read_csv(os.path.join(base, 'data/processed/streaming_users_clean.csv'),
                       parse_dates=['last_login_date'])

df = load_data()

# Calcular valores reales
orden = ['Básico', 'Estándar', 'Premium']
medias = df.groupby('subscription_plan')['monthly_watch_time_mins'].mean().reindex(orden)
corr_tickets = df['customer_support_tickets'].corr(df['monthly_watch_time_mins'])

num_cols = ['age', 'monthly_watch_time_mins', 'customer_support_tickets']
X_imp = SimpleImputer(strategy='mean').fit_transform(
    StandardScaler().fit_transform(df[num_cols]))
pca = PCA()
pca.fit(X_imp)
varianza = pca.explained_variance_ratio_
varianza_acum = np.cumsum(varianza)
n_80 = np.argmax(varianza_acum >= 0.80) + 1

st.subheader("Respuesta a las preguntas de análisis")

# P1
st.markdown("### P1 — ¿Existe relación entre el plan de suscripción y el tiempo de visualización?")
col1, col2, col3 = st.columns(3)
col1.metric("Básico", f"{medias['Básico']:.0f} min/mes")
col2.metric("Estándar", f"{medias['Estándar']:.0f} min/mes")
col3.metric("Premium", f"{medias['Premium']:.0f} min/mes")
st.success(f"**Conclusión P1:** Sí existe una relación positiva y gradual. Los usuarios Premium "
           f"consumen casi el doble que los Básicos (≈{medias['Premium']:.0f} vs. "
           f"≈{medias['Básico']:.0f} min/mes), una diferencia de "
           f"≈{medias['Premium']-medias['Básico']:.0f} minutos mensuales. "
           "El plan es el principal diferenciador del nivel de consumo.")

st.markdown("---")

# P2
st.markdown("### P2 — ¿Cuál es la distribución de usuarios por país y género?")
col1, col2 = st.columns(2)
with col1:
    st.write("**Por país:**")
    pais = df['country'].value_counts().reset_index()
    pais.columns = ['País', 'Usuarios']
    pais['%'] = (pais['Usuarios'] / len(df) * 100).round(1)
    st.dataframe(pais, use_container_width=True, hide_index=True)
with col2:
    st.write("**Por género favorito:**")
    gen = df['favorite_genre'].value_counts().reset_index()
    gen.columns = ['Género', 'Usuarios']
    gen['%'] = (gen['Usuarios'] / len(df) * 100).round(1)
    st.dataframe(gen, use_container_width=True, hide_index=True)
st.success("**Conclusión P2:** La distribución geográfica es homogénea entre los 7 países, "
           "sin concentración dominante en un único mercado. Las preferencias de contenido "
           "se concentran en Drama, Acción y Comedia.")

st.markdown("---")

# P3
st.markdown("### P3 — ¿Qué variables numéricas presentan mayor correlación?")
corr_df = df[num_cols].corr().round(3)
st.dataframe(corr_df, use_container_width=True)
st.success("**Conclusión P3:** Las correlaciones entre las tres variables numéricas son "
           "prácticamente nulas (r < 0.01 en todos los pares). Las variables son "
           "estadísticamente independientes entre sí, sin multicolinealidad.")

st.markdown("---")

# P4
st.markdown("### P4 — ¿Qué estructura latente revela el PCA?")
col1, col2, col3 = st.columns(3)
col1.metric("PC1", f"{varianza[0]*100:.1f}%")
col2.metric("PC2", f"{varianza[1]*100:.1f}%")
col3.metric("PC3", f"{varianza[2]*100:.1f}%")
st.success(f"**Conclusión P4:** Se necesitan las {n_80} componentes para superar el 80% de "
           "varianza acumulada, confirmando que las variables son independientes. "
           "PC1 combina edad y consumo (compromiso general), PC2 contrapone edad y consumo, "
           "PC3 está dominada por tickets de soporte. El PCA aporta valor interpretativo, "
           "no de compresión.")

st.markdown("---")

# P5
st.markdown("### P5 — ¿Los usuarios con más tickets consumen menos?")
st.metric("Correlación tickets vs. tiempo de visualización", f"r = {corr_tickets:.3f}")
st.success(f"**Conclusión P5:** La correlación r = {corr_tickets:.3f} refuta la hipótesis. "
           "Los tickets de soporte no se asocian con menor consumo. No deben usarse como "
           "proxy aislado de insatisfacción o riesgo de abandono.")

st.markdown("---")
st.subheader("Limitaciones")
st.warning("""
- El dataset no incluye información sobre el contenido consumido ni la fecha de alta del usuario,
  lo que limita el análisis de comportamiento temporal.
- `last_login_date` conserva valores NaT sin imputar, limitando análisis de recencia.
- Las correlaciones analizadas son lineales (Pearson); no se exploraron relaciones no lineales.
- La independencia entre variables podría deberse en parte a la naturaleza sintética del dataset.
""")

st.markdown("---")
st.subheader("Próximos pasos")
st.info("""
- Incorporar variables de contenido (títulos vistos, minutos por género) para profundizar
  el análisis de preferencias.
- Aplicar clustering (K-Means) sobre los componentes del PCA para segmentar perfiles de usuario.
- Integrar datos longitudinales para analizar la evolución del comportamiento a lo largo del tiempo.
""")

st.markdown("---")
col1, col2 = st.columns(2)
col1.markdown("🔗 [Repositorio GitHub](https://github.com/MajoAchaval/PI_Mineria_Datos_1)")
col2.markdown("🌐 Aplicación en Streamlit Cloud — enlace disponible tras el deploy")