import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# Estilo profesional global
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'axes.grid': True,
    'grid.color': '#E0E0E0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'font.family': 'sans-serif',
    'axes.labelcolor': '#333333',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
})

AZUL     = '#1D4ED8'
VIOLETA  = '#7C3AED'
NARANJA  = '#F97316'
VERDE    = '#059669'
CORAL    = '#F43F5E'
GRIS     = '#6B7280'
PALETA   = {'Básico': '#3B82F6', 'Estándar': '#F97316', 'Premium': '#059669'}

st.title("📊 Análisis Exploratorio de Datos")
st.caption("Explorá los patrones de comportamiento de los usuarios filtrando por país y plan.")

@st.cache_data
def load_data():
    base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    return pd.read_csv(os.path.join(base, 'data/processed/streaming_users_clean.csv'),
                       parse_dates=['last_login_date'])

df = load_data()

# Filtros
st.sidebar.header("🔧 Filtros")
paises = st.sidebar.multiselect("País", sorted(df['country'].dropna().unique()),
                                 default=sorted(df['country'].dropna().unique()))
planes = st.sidebar.multiselect("Plan", ['Básico', 'Estándar', 'Premium'],
                                 default=['Básico', 'Estándar', 'Premium'])
df_f = df[df['country'].isin(paises) & df['subscription_plan'].isin(planes)]
st.sidebar.metric("Registros filtrados", f"{len(df_f):,}")

st.markdown("---")

# ── UV-1 ─────────────────────────────────────────────────────────────────────
st.subheader("UV-1 — Distribución de edades")
fig, ax = plt.subplots(figsize=(14, 5))
ax.hist(df_f['age'], bins=30, color=AZUL, edgecolor='white', alpha=0.85, linewidth=0.5)
ax.axvline(df_f['age'].mean(), color='#DC2626', linestyle='--', linewidth=1.5,
           label=f"Media: {df_f['age'].mean():.1f} años")
ax.axvline(df_f['age'].median(), color=NARANJA, linestyle='--', linewidth=1.5,
           label=f"Mediana: {df_f['age'].median():.1f} años")
ax.set_xlabel("Edad", fontsize=11)
ax.set_ylabel("Cantidad de usuarios", fontsize=11)
ax.legend(frameon=False, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.info(f"La distribución de edades se concentra entre los 25 y 45 años, con media "
        f"{df_f['age'].mean():.1f} y mediana {df_f['age'].median():.1f}. "
        "La proximidad entre ambos valores indica una distribución aproximadamente simétrica, "
        "sin sesgo marcado hacia usuarios muy jóvenes o muy mayores.")

st.markdown("---")

# ── UV-2 ─────────────────────────────────────────────────────────────────────
st.subheader("UV-2 — Distribución del tiempo mensual de visualización")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(df_f['monthly_watch_time_mins'], bins=40, color=VIOLETA,
             edgecolor='white', alpha=0.85, linewidth=0.5, density=True)

from scipy.stats import gaussian_kde
import numpy as np
datos = df_f['monthly_watch_time_mins'].dropna()
kde = gaussian_kde(datos)
x_range = np.linspace(datos.min(), datos.max(), 300)
axes[0].plot(x_range, kde(x_range), color='white', linewidth=3.5, zorder=4)
axes[0].plot(x_range, kde(x_range), color='#1E1B4B', linewidth=2, zorder=5)
axes[0].set_xlabel("Minutos / mes", fontsize=11)
axes[0].set_ylabel("Cantidad de usuarios", fontsize=11)
axes[0].set_title("Histograma", fontsize=11, color="#854545")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

bp = axes[1].boxplot(df_f['monthly_watch_time_mins'].dropna(), vert=False,
                     patch_artist=True, widths=0.5,
                     boxprops=dict(facecolor='#DBEAFE', color=AZUL, linewidth=1.5),
                     medianprops=dict(color='#DC2626', linewidth=2),
                     whiskerprops=dict(color=AZUL, linewidth=1.2),
                     capprops=dict(color=AZUL, linewidth=1.2),
                     flierprops=dict(marker='o', color=GRIS, alpha=0.3, markersize=3))
axes[1].set_xlabel("Minutos / mes", fontsize=11)
axes[1].set_title("Boxplot", fontsize=11, color='#333333')
axes[1].set_yticks([])

plt.tight_layout()
st.pyplot(fig)
plt.close()
st.info(f"El tiempo de visualización presenta una distribución asimétrica a la derecha, "
        f"con media ≈ {df_f['monthly_watch_time_mins'].mean():.0f} min/mes y "
        f"mediana ≈ {df_f['monthly_watch_time_mins'].median():.0f} min/mes. "
        "El boxplot confirma que los valores extremos residuales son acotados gracias "
        "a la winsorización aplicada en el proceso de limpieza.")

st.markdown("---")

# ── BV-1 ─────────────────────────────────────────────────────────────────────
st.subheader("BV-1 — Tiempo de visualización por plan de suscripción")
orden = [p for p in ['Básico', 'Estándar', 'Premium'] if p in df_f['subscription_plan'].unique()]
medias = df_f.groupby('subscription_plan')['monthly_watch_time_mins'].mean().reindex(orden)
colores = [PALETA[p] for p in orden]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(orden, medias, color=colores, edgecolor='white', linewidth=0.5,
              width=0.5, zorder=3)
for bar, val in zip(bars, medias):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:.0f} min', ha='center', fontsize=11,
            fontweight='bold', color='#333333')
ax.set_xlabel("Plan de suscripción", fontsize=11)
ax.set_ylabel("Minutos promedio / mes", fontsize=11)
ax.set_ylim(0, medias.max() * 1.18)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.info("Los usuarios Premium consumen casi el doble que los usuarios Básicos "
        "(≈1.123 vs. ≈587 min/mes). Esta relación escalonada y consistente indica que "
        "el plan de suscripción es el principal diferenciador del nivel de consumo.")

st.markdown("---")

# ── BV-2 ─────────────────────────────────────────────────────────────────────
st.subheader("BV-2 — Tickets de soporte vs. tiempo de visualización")
corr = df_f['customer_support_tickets'].corr(df_f['monthly_watch_time_mins'])

fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(df_f['customer_support_tickets'], df_f['monthly_watch_time_mins'],
           alpha=0.35, s=16, color=CORAL, edgecolors='none', zorder=3)
ax.set_xlabel("Tickets de soporte", fontsize=11)
ax.set_ylabel("Minutos / mes", fontsize=11)
ax.text(0.97, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#D1D5DB', alpha=0.9))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.info(f"La correlación r = {corr:.3f} es prácticamente nula, lo que indica que los "
        "usuarios con más tickets no consumen menos. El consumo se mantiene estable "
        "independientemente de la cantidad de tickets de soporte generados.")

st.markdown("---")

# ── MV-1 ─────────────────────────────────────────────────────────────────────
st.subheader("MV-1 — Tiempo de visualización por país y plan de suscripción")
orden_planes = [p for p in ['Básico', 'Estándar', 'Premium']
                if p in df_f['subscription_plan'].unique()]
pivot = (df_f.groupby(['country', 'subscription_plan'])['monthly_watch_time_mins']
         .mean().unstack().reindex(columns=orden_planes))

fig, ax = plt.subplots(figsize=(16, 6))
x = range(len(pivot))
width = 0.25
for i, plan in enumerate(orden_planes):
    offset = (i - 1) * width
    bars = ax.bar([xi + offset for xi in x], pivot[plan],
                  width=width, label=plan, color=PALETA[plan],
                  edgecolor='white', linewidth=0.5, zorder=3)

ax.set_xticks(list(x))
ax.set_xticklabels(pivot.index, rotation=30, ha='right', fontsize=10)
ax.set_xlabel("País", fontsize=11)
ax.set_ylabel("Minutos promedio / mes", fontsize=11)
ax.legend(title="Plan", frameon=False, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.info("El patrón Premium > Estándar > Básico se mantiene consistente en todos los países "
        "analizados, lo que indica que la relación entre plan y consumo no varía "
        "significativamente por mercado geográfico.")