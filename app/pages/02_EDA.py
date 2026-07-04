import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.title("📊 Análisis Exploratorio de Datos")
st.caption("Explorá los patrones de comportamiento de los usuarios filtrando por país y plan.")

@st.cache_data
def load_data():
    base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    return pd.read_csv(os.path.join(base, 'data/processed/streaming_users_clean.csv'),
                       parse_dates=['last_login_date'])

df = load_data()

PALETA = {'Básico': '#3B82F6', 'Estándar': '#F97316', 'Premium': '#059669'}

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
fig = px.histogram(df_f, x='age', nbins=30,
                   color_discrete_sequence=['#3B82F6'],
                   marginal='violin',
                   opacity=0.85,
                   labels={'age': 'Edad', 'count': 'Usuarios'})
fig.add_vline(x=df_f['age'].mean(), line_dash='dash', line_color='#DC2626',
              annotation_text=f"Media: {df_f['age'].mean():.1f}",
              annotation_position='top right')
fig.add_vline(x=df_f['age'].median(), line_dash='dash', line_color='#F97316',
              annotation_text=f"Mediana: {df_f['age'].median():.1f}",
              annotation_position='top left')
fig.update_layout(showlegend=False, height=420,
                  plot_bgcolor='white', paper_bgcolor='white',
                  xaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                  yaxis=dict(showgrid=True, gridcolor='#F0F0F0', title='Usuarios'))
st.plotly_chart(fig, use_container_width=True)
st.info(f"La distribución de edades se concentra entre los 25 y 45 años, con media "
        f"{df_f['age'].mean():.1f} y mediana {df_f['age'].median():.1f}. "
        "La proximidad entre ambos valores indica una distribución aproximadamente simétrica.")

st.markdown("---")

# ── UV-2 ─────────────────────────────────────────────────────────────────────
st.subheader("UV-2 — Distribución del tiempo mensual de visualización")
fig = px.histogram(df_f, x='monthly_watch_time_mins', nbins=40,
                   color_discrete_sequence=['#7C3AED'],
                   marginal='box',
                   opacity=0.85,
                   labels={'monthly_watch_time_mins': 'Minutos / mes', 'count': 'Usuarios'})
fig.add_vline(x=df_f['monthly_watch_time_mins'].mean(), line_dash='dash',
              line_color='#DC2626',
              annotation_text=f"Media: {df_f['monthly_watch_time_mins'].mean():.0f} min",
              annotation_position='top right')
fig.add_vline(x=df_f['monthly_watch_time_mins'].median(), line_dash='dash',
              line_color='#F97316',
              annotation_text=f"Mediana: {df_f['monthly_watch_time_mins'].median():.0f} min",
              annotation_position='top left')
fig.update_layout(showlegend=False, height=420,
                  plot_bgcolor='white', paper_bgcolor='white',
                  xaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                  yaxis=dict(showgrid=True, gridcolor='#F0F0F0', title='Usuarios'))
st.plotly_chart(fig, use_container_width=True)
st.info(f"El tiempo de visualización presenta una distribución asimétrica a la derecha, "
        f"con media ≈ {df_f['monthly_watch_time_mins'].mean():.0f} min/mes y "
        f"mediana ≈ {df_f['monthly_watch_time_mins'].median():.0f} min/mes.")

st.markdown("---")

# ── BV-1 ─────────────────────────────────────────────────────────────────────
st.subheader("BV-1 — Tiempo de visualización por plan de suscripción")
orden = [p for p in ['Básico', 'Estándar', 'Premium'] if p in df_f['subscription_plan'].unique()]
medias = df_f.groupby('subscription_plan')['monthly_watch_time_mins'].mean().reindex(orden).reset_index()
medias.columns = ['Plan', 'Minutos promedio']

fig = px.bar(medias, x='Plan', y='Minutos promedio',
             color='Plan', color_discrete_map=PALETA,
             text='Minutos promedio',
             labels={'Minutos promedio': 'Minutos promedio / mes'})
fig.update_traces(texttemplate='%{text:.0f} min', textposition='outside',
                  marker_line_width=0)
fig.update_layout(showlegend=False, height=420,
                  plot_bgcolor='white', paper_bgcolor='white',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                  uniformtext_minsize=10)
st.plotly_chart(fig, use_container_width=True)
st.info("Los usuarios Premium consumen casi el doble que los usuarios Básicos "
        "(≈1.123 vs. ≈587 min/mes). El plan de suscripción es el principal "
        "diferenciador del nivel de consumo.")

st.markdown("---")

# ── BV-2 ─────────────────────────────────────────────────────────────────────
st.subheader("BV-2 — Tickets de soporte vs. tiempo de visualización")
corr = df_f['customer_support_tickets'].corr(df_f['monthly_watch_time_mins'])
fig = px.scatter(df_f, x='customer_support_tickets', y='monthly_watch_time_mins',
                 opacity=0.35, color_discrete_sequence=['#F43F5E'],
                 labels={'customer_support_tickets': 'Tickets de soporte',
                         'monthly_watch_time_mins': 'Minutos / mes'},
                 trendline='ols')
fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
fig.update_layout(height=420,
                  plot_bgcolor='white', paper_bgcolor='white',
                  xaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                  yaxis=dict(showgrid=True, gridcolor='#F0F0F0'))
fig.add_annotation(x=0.97, y=0.97, xref='paper', yref='paper',
                   text=f'r = {corr:.3f}', showarrow=False,
                   bgcolor='white', bordercolor='#D1D5DB',
                   font=dict(size=13))
st.plotly_chart(fig, use_container_width=True)
st.info(f"La correlación r = {corr:.3f} es prácticamente nula. El consumo se mantiene "
        "estable independientemente de la cantidad de tickets de soporte generados.")

st.markdown("---")

# ── MV-1 ─────────────────────────────────────────────────────────────────────
st.subheader("MV-1 — Tiempo de visualización por país y plan de suscripción")
orden_planes = [p for p in ['Básico', 'Estándar', 'Premium']
                if p in df_f['subscription_plan'].unique()]
mv1 = (df_f.groupby(['country', 'subscription_plan'])['monthly_watch_time_mins']
       .mean().round(0).reset_index())
mv1.columns = ['País', 'Plan', 'Minutos promedio']

fig = px.bar(mv1, x='País', y='Minutos promedio', color='Plan',
             color_discrete_map=PALETA, barmode='group',
             labels={'Minutos promedio': 'Minutos promedio / mes'},
             category_orders={'Plan': orden_planes})
fig.update_traces(marker_line_width=0)
fig.update_layout(height=480,
                  plot_bgcolor='white', paper_bgcolor='white',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                  legend=dict(title='Plan', orientation='h',
                              yanchor='bottom', y=1.02,
                              xanchor='right', x=1))
st.plotly_chart(fig, use_container_width=True)
st.info("El patrón Premium > Estándar > Básico se mantiene consistente en todos los países, "
        "lo que indica que la relación entre plan y consumo no varía por mercado geográfico.")