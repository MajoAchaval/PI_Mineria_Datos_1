import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from pathlib import Path

st.title("📝 Conclusiones")
st.caption("Hallazgos principales del análisis de usuarios de la plataforma de streaming.")

@st.cache_data
def load_data():
    base = Path(__file__).resolve().parents[2]
    return pd.read_csv(base / 'data/processed/streaming_users_clean.csv',
                       parse_dates=['last_login_date'])

df = load_data()

orden = ['Básico', 'Estándar', 'Premium']
medias = df.groupby('subscription_plan')['monthly_watch_time_mins'].mean().reindex(orden)
corr_tickets = df['customer_support_tickets'].corr(df['monthly_watch_time_mins'])

num_cols = ['age', 'monthly_watch_time_mins', 'customer_support_tickets']
X_imp = SimpleImputer(strategy='mean').fit_transform(
    StandardScaler().fit_transform(df[num_cols]))
pca = PCA()
pca.fit(X_imp)
varianza = pca.explained_variance_ratio_

st.markdown("---")

# ── HALLAZGO 1 ───────────────────────────────────────────────────────────────
st.subheader("🏆 El plan define cuánto consume cada usuario")
st.write("Los usuarios Premium ven casi el **doble** de contenido que los usuarios Básicos cada mes.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("👤 Básico", f"{medias['Básico']:.0f} min/mes")
col2.metric("👤 Estándar", f"{medias['Estándar']:.0f} min/mes")
col3.metric("👤 Premium", f"{medias['Premium']:.0f} min/mes")
col4.metric("📈 Diferencia",
            f"+{medias['Premium']-medias['Básico']:.0f} min/mes",
            delta_color="normal")
st.success("El plan de suscripción es el factor que mejor explica el nivel de consumo, "
           "y este patrón se repite de forma consistente en los 7 países analizados.")

st.markdown("---")

# ── HALLAZGO 2 ───────────────────────────────────────────────────────────────
st.subheader("👤 La plataforma tiene una base de usuarios adulta joven")
st.write("La mayoría de los usuarios tiene entre **25 y 45 años**, "
         "con una distribución aproximadamente simétrica.")

col1, col2, col3 = st.columns(3)
col1.metric("Edad promedio", f"{df['age'].mean():.1f} años")
col2.metric("Edad mediana", f"{df['age'].median():.1f} años")
col3.metric("Rango principal", "25 — 45 años")

st.success("El perfil etario dominante es el adulto joven. No hay sesgo marcado "
           "hacia usuarios muy jóvenes ni muy mayores.")

st.markdown("---")

# ── HALLAZGO 3 ───────────────────────────────────────────────────────────────
st.subheader("🎫 Más tickets de soporte no significa menos consumo")
st.write("Contra lo esperado, los usuarios que contactan más al soporte técnico "
         "no consumen menos contenido.")

col1, col2 = st.columns(2)
with col1:
    avg_by_tickets = df.groupby('customer_support_tickets')['monthly_watch_time_mins'].mean()
    st.dataframe(
        avg_by_tickets.reset_index().rename(columns={
            'customer_support_tickets': 'Tickets de soporte',
            'monthly_watch_time_mins': 'Promedio min/mes'
        }).round(0),
        use_container_width=True, hide_index=True)
with col2:
    st.metric("Relación entre tickets y consumo", "Sin correlación",
              delta=f"r ≈ {corr_tickets:.3f}", delta_color="off")
    st.write("El tiempo de visualización se mantiene estable sin importar "
             "la cantidad de tickets generados.")
st.success("Los tickets de soporte no son un indicador confiable de insatisfacción "
           "ni de riesgo de abandono de la plataforma.")

st.markdown("---")

# ── HALLAZGO 4 ───────────────────────────────────────────────────────────────
st.subheader("🔬 Cada variable describe una dimensión distinta del usuario")
st.write("El análisis estadístico confirmó que la edad, el consumo y los tickets "
         "de soporte son aspectos **independientes** entre sí.")

col1, col2, col3 = st.columns(3)
col1.metric("PC1", f"{varianza[0]*100:.1f}%")
col2.metric("PC2", f"{varianza[1]*100:.1f}%")
col3.metric("PC3", f"{varianza[2]*100:.1f}%")

col1, col2, col3 = st.columns(3)
col1.info("**👤 Edad**\nNo predice cuánto consume ni cuántos tickets genera un usuario.")
col2.info("**📺 Consumo mensual**\nNo está relacionado con la edad ni con el soporte.")
col3.info("**🎫 Tickets de soporte**\nNo depende del plan ni del tiempo de visualización.")

st.success("Cada variable aporta información única e independiente. "
           "Para entender a un usuario en profundidad es necesario "
           "considerar las tres dimensiones juntas.")

st.markdown("---")

# ── LIMITACIONES ─────────────────────────────────────────────────────────────
st.subheader("⚠️ Limitaciones")
st.warning(
    "El alcance de las conclusiones está condicionado por la información disponible "
    "y las decisiones documentadas durante el proceso. El dataset no incluye información "
    "sobre el contenido consumido ni sobre la fecha de alta del usuario, lo que restringe "
    "el análisis al perfil estático del usuario en un momento dado. "
    "La naturaleza sintética del dataset limita la generalización a datos reales "
    "de plataformas de streaming.")

st.markdown("---")

# ── MEJORAS FUTURAS ───────────────────────────────────────────────────────────
st.subheader("🚀 ¿Qué se podría hacer a futuro?")
st.info(
    "Una mejora futura podría consistir en incorporar variables de contenido "
    "(títulos vistos y minutos por género), aplicar clustering para segmentar perfiles "
    "de usuario, e integrar datos longitudinales para analizar la evolución del "
    "comportamiento a lo largo del tiempo.")

st.markdown("---")
col1, col2 = st.columns(2)
col1.markdown("🔗 [Repositorio GitHub](https://github.com/MajoAchaval/PI_Mineria_Datos_1)")
col2.markdown("🌐 [Aplicación en Streamlit Cloud](https://pi-mineriadatos1.streamlit.app/)")