import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("📊 Análisis Exploratorio de Datos")

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, 'data/processed/streaming_users_clean.csv'),
                       parse_dates=['last_login_date'])

df = load_data()

# Filtros en sidebar
st.sidebar.header("Filtros")
paises = st.sidebar.multiselect("País", sorted(df['country'].dropna().unique()),
                                 default=sorted(df['country'].dropna().unique()))
planes = st.sidebar.multiselect("Plan", ['Básico', 'Estándar', 'Premium'],
                                 default=['Básico', 'Estándar', 'Premium'])
df_f = df[df['country'].isin(paises) & df['subscription_plan'].isin(planes)]
st.sidebar.metric("Registros filtrados", f"{len(df_f):,}")

st.markdown("---")

# UV-1
st.subheader("🔵 UV-1: Distribución de edades")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_f['age'], bins=30, color='steelblue', edgecolor='white')
ax.axvline(df_f['age'].mean(), color='red', linestyle='--',
           label=f"Media: {df_f['age'].mean():.1f}")
ax.axvline(df_f['age'].median(), color='orange', linestyle='--',
           label=f"Mediana: {df_f['age'].median():.1f}")
ax.set_xlabel("Edad")
ax.set_ylabel("Usuarios")
ax.legend()
st.pyplot(fig)
st.info(f"**Interpretación:** La distribución de edades se concentra entre los 25 y 45 años, "
        f"con media {df_f['age'].mean():.1f} y mediana {df_f['age'].median():.1f}. "
        "La proximidad entre ambos valores indica una distribución aproximadamente simétrica, "
        "sin sesgo marcado hacia usuarios muy jóvenes o muy mayores.")

st.markdown("---")

# UV-2
st.subheader("🔵 UV-2: Distribución del tiempo mensual de visualización")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_f['monthly_watch_time_mins'], bins=40, color='teal', edgecolor='white')
axes[0].set_xlabel("Minutos / mes")
axes[0].set_ylabel("Usuarios")
axes[0].set_title("Histograma")
axes[1].boxplot(df_f['monthly_watch_time_mins'], vert=False)
axes[1].set_xlabel("Minutos / mes")
axes[1].set_title("Boxplot")
plt.tight_layout()
st.pyplot(fig)
st.info(f"**Interpretación:** El tiempo de visualización presenta una distribución asimétrica a la derecha, "
        f"con media ≈ {df_f['monthly_watch_time_mins'].mean():.0f} min/mes y "
        f"mediana ≈ {df_f['monthly_watch_time_mins'].median():.0f} min/mes. "
        "El boxplot confirma que los valores extremos residuales post-limpieza son acotados "
        "gracias a la winsorización aplicada en el proceso de limpieza.")

st.markdown("---")

# BV-1
st.subheader("🟠 BV-1: Tiempo de visualización por plan de suscripción")
orden = [p for p in ['Básico', 'Estándar', 'Premium'] if p in df_f['subscription_plan'].unique()]
medias = df_f.groupby('subscription_plan')['monthly_watch_time_mins'].mean().reindex(orden)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(orden, medias, color=['#5B9BD5', '#ED7D31', '#A9D18E'])
ax.set_xlabel("Plan")
ax.set_ylabel("Minutos promedio / mes")
for bar, val in zip(bars, medias):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.0f}', ha='center', fontweight='bold')
st.pyplot(fig)
st.info("**Interpretación:** Los usuarios Premium consumen casi el doble que los usuarios Básico "
        "(≈1.123 vs. ≈587 min/mes). Esta relación escalonada y consistente indica que el plan "
        "de suscripción es el principal diferenciador del nivel de consumo en la plataforma.")

st.markdown("---")

# BV-2
st.subheader("🟠 BV-2: Tickets de soporte vs. tiempo de visualización")
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(df_f['customer_support_tickets'], df_f['monthly_watch_time_mins'],
           alpha=0.3, s=12, color='coral')
corr = df_f['customer_support_tickets'].corr(df_f['monthly_watch_time_mins'])
ax.set_xlabel("Tickets de soporte")
ax.set_ylabel("Minutos / mes")
ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.3))
st.pyplot(fig)
st.info(f"**Interpretación:** La correlación r = {corr:.3f} es prácticamente nula, lo que refuta "
        "la hipótesis de que los usuarios con más problemas técnicos consumen menos. "
        "El consumo se mantiene estable independientemente de la cantidad de tickets generados.")

st.markdown("---")

# MV-1
st.subheader("🟣 MV-1: Tiempo de visualización por país y plan de suscripción")
orden_planes = [p for p in ['Básico', 'Estándar', 'Premium'] if p in df_f['subscription_plan'].unique()]
pivot = df_f.groupby(['country', 'subscription_plan'])['monthly_watch_time_mins'].mean().unstack()
pivot = pivot.reindex(columns=orden_planes)
fig, ax = plt.subplots(figsize=(12, 5))
pivot.plot(kind='bar', ax=ax, color=['#5B9BD5', '#ED7D31', '#A9D18E'], edgecolor='white')
ax.set_xlabel("País")
ax.set_ylabel("Minutos promedio / mes")
ax.set_title("Tiempo de visualización por país y plan")
ax.legend(title="Plan")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
st.info("**Interpretación:** El patrón Premium > Estándar > Básico se mantiene consistente en todos "
        "los países analizados, lo que indica que la relación entre plan y consumo no varía "
        "significativamente por mercado geográfico.")