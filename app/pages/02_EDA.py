import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Análisis Exploratorio de Datos")

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/streaming_users_clean.csv', parse_dates=['last_login_date'])

df = load_data()

# Filtros opcionales en sidebar
st.sidebar.header("Filtros")
paises = st.sidebar.multiselect("País", sorted(df['country'].dropna().unique()),
                                 default=sorted(df['country'].dropna().unique()))
planes = st.sidebar.multiselect("Plan", ['Básico', 'Estándar', 'Premium'],
                                 default=['Básico', 'Estándar', 'Premium'])
df_f = df[df['country'].isin(paises) & df['subscription_plan'].isin(planes)]
st.sidebar.metric("Registros filtrados", f"{len(df_f):,}")

st.markdown("---")

# UV-1: Distribución de edades
st.subheader("🔵 UV-1: Distribución de edades")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_f['age'], bins=30, color='steelblue', edgecolor='white')
ax.axvline(df_f['age'].mean(), color='red', linestyle='--', label=f"Media: {df_f['age'].mean():.1f}")
ax.axvline(df_f['age'].median(), color='orange', linestyle='--', label=f"Mediana: {df_f['age'].median():.1f}")
ax.set_xlabel("Edad"); ax.set_ylabel("Usuarios"); ax.legend()
st.pyplot(fig)
st.info("**Interpretación:** La distribución de edades permite identificar el perfil demográfico dominante. "
        "La diferencia entre media y mediana indica el grado de asimetría de la distribución. "
        "Completar con los valores reales observados al ejecutar.")

st.markdown("---")

# UV-2: Tiempo de visualización
st.subheader("🔵 UV-2: Distribución del tiempo mensual de visualización")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_f['monthly_watch_time_mins'], bins=40, color='teal', edgecolor='white')
axes[0].set_xlabel("Minutos / mes"); axes[0].set_ylabel("Usuarios")
axes[0].set_title("Histograma")
axes[1].boxplot(df_f['monthly_watch_time_mins'], vert=False)
axes[1].set_xlabel("Minutos / mes")
axes[1].set_title("Boxplot")
plt.tight_layout()
st.pyplot(fig)
st.info("**Interpretación:** El boxplot permite identificar la dispersión central y los valores extremos "
        "residuales post-limpieza. Completar con los valores reales observados.")

st.markdown("---")

# BV-1: Tiempo por plan
st.subheader("🟠 BV-1: Tiempo de visualización por plan de suscripción")
orden = [p for p in ['Básico', 'Estándar', 'Premium'] if p in df_f['subscription_plan'].unique()]
medias = df_f.groupby('subscription_plan')['monthly_watch_time_mins'].mean().reindex(orden)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(orden, medias, color=['#5B9BD5', '#ED7D31', '#A9D18E'])
ax.set_xlabel("Plan"); ax.set_ylabel("Minutos promedio / mes")
for bar, val in zip(bars, medias):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', ha='center')
st.pyplot(fig)
st.info("**Interpretación:** Esta visualización permite evaluar si el plan de suscripción está asociado "
        "con el nivel de consumo. Una diferencia significativa entre planes sugiere que los usuarios con "
        "planes superiores aprovechan más el servicio. Completar con valores reales.")

st.markdown("---")

# BV-2: Tickets vs. watch time
st.subheader("🟠 BV-2: Tickets de soporte vs. tiempo de visualización")
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(df_f['customer_support_tickets'], df_f['monthly_watch_time_mins'],
           alpha=0.3, s=12, color='coral')
corr = df_f['customer_support_tickets'].corr(df_f['monthly_watch_time_mins'])
ax.set_xlabel("Tickets de soporte"); ax.set_ylabel("Minutos / mes")
ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.3))
st.pyplot(fig)
st.info(f"**Interpretación:** La correlación de Pearson r = {corr:.3f} entre tickets de soporte y "
        "tiempo de visualización indica la magnitud y dirección de la relación. "
        "Completar con la interpretación sustantiva.")

st.markdown("---")

# MV-1: Heatmap de correlaciones
st.subheader("🟣 MV-1: Mapa de correlaciones entre variables numéricas")
num_cols = ['age', 'monthly_watch_time_mins', 'customer_support_tickets']
corr_m = df_f[num_cols].corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr_m, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax,
            linewidths=0.5, square=True)
st.pyplot(fig)
st.info("**Interpretación:** El mapa de correlaciones revela la estructura de dependencia lineal entre las "
        "variables numéricas del dataset. Las correlaciones próximas a 0 indican independencia entre "
        "dimensiones, lo que es relevante para el análisis de PCA.")
