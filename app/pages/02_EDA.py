import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.title("📊 Análisis Exploratorio de Datos")
st.caption("Explorá los patrones de comportamiento de los usuarios filtrando por país y plan.")

@st.cache_data
def load_data():
    base = Path(__file__).resolve().parents[2]
    return pd.read_csv(base / 'data/processed/streaming_users_clean.csv',
                       parse_dates=['last_login_date'])

df = load_data()

PALETA = {'Básico': '#6366F1', 'Estándar': '#F59E0B', 'Premium': '#10B981'}

def base_layout(height=420):
    return dict(
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=12, color='#374151'),
        margin=dict(t=50, b=50, l=50, r=30),
        xaxis=dict(showgrid=False, linecolor='#E5E7EB', linewidth=1,
                   tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6', linecolor='#E5E7EB',
                   linewidth=1, tickfont=dict(size=11)),
        hoverlabel=dict(bgcolor='white', bordercolor='#E5E7EB',
                        font=dict(size=12, color='#374151')),
    )

LAYOUT = base_layout()

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

media_age = df_f['age'].mean()
mediana_age = df_f['age'].median()

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=df_f['age'], nbinsx=30,
    marker=dict(color='#6366F1', line=dict(color='white', width=0.8)),
    opacity=0.9, name='Usuarios', hovertemplate='Edad: %{x}<br>Usuarios: %{y}<extra></extra>'
))
fig.add_vline(x=media_age, line_dash='dash', line_color='#EF4444', line_width=2,
              annotation_text=f"Media: {media_age:.1f} años",
              annotation_position='top right',
              annotation=dict(font=dict(color='#EF4444', size=11)))
fig.add_vline(x=mediana_age, line_dash='dot', line_color='#F59E0B', line_width=2,
              annotation_text=f"Mediana: {mediana_age:.1f} años",
              annotation_position='top left',
              annotation=dict(font=dict(color='#F59E0B', size=11)))
fig.update_layout(**base_layout(),
                  xaxis_title='Edad',
                  yaxis_title='Cantidad de usuarios',
                  showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.info(f"La distribución de edades se concentra entre los 25 y 45 años, con media "
        f"{media_age:.1f} y mediana {mediana_age:.1f}. "
        "La distribución es aproximadamente simétrica, sin sesgo marcado hacia "
        "usuarios muy jóvenes o muy mayores.")

st.markdown("---")

# ── UV-2 ─────────────────────────────────────────────────────────────────────
st.subheader("UV-2 — Distribución por género favorito")

generos = df_f['favorite_genre'].value_counts().reset_index()
generos.columns = ['Género', 'Usuarios']
generos['Porcentaje'] = (generos['Usuarios'] / len(df_f) * 100).round(1)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=generos['Usuarios'],
    y=generos['Género'],
    orientation='h',
    marker=dict(
        color=generos['Usuarios'],
        colorscale=[[0, '#C4B5FD'], [1, '#6366F1']],
        line=dict(color='white', width=0)
    ),
    text=[f"{p}%" for p in generos['Porcentaje']],
    textposition='outside',
    textfont=dict(size=11, color='#374151'),
    hovertemplate='%{y}: %{x:,} usuarios<extra></extra>',
))
fig.update_layout(**LAYOUT, height=420, showlegend=False)
fig.update_xaxes(showgrid=True, gridcolor='#F3F4F6',
                 linecolor='#E5E7EB', title='Cantidad de usuarios')
fig.update_yaxes(showgrid=False, linecolor='#E5E7EB', autorange='reversed')
st.plotly_chart(fig, use_container_width=True)
st.info("Drama, Acción y Comedia concentran las preferencias de contenido en la plataforma. "
        "Esta distribución es consistente en todos los países analizados y orienta "
        "la política de catálogo hacia estos tres géneros prioritarios.")

st.markdown("---")

# ── BV-1 ─────────────────────────────────────────────────────────────────────
st.subheader("BV-1 — Tiempo de visualización por plan de suscripción")

orden = [p for p in ['Básico', 'Estándar', 'Premium']
         if p in df_f['subscription_plan'].unique()]
medias = (df_f.groupby('subscription_plan')['monthly_watch_time_mins']
          .mean().reindex(orden).reset_index())
medias.columns = ['Plan', 'Minutos']

fig = go.Figure()
for _, row in medias.iterrows():
    fig.add_trace(go.Bar(
        x=[row['Plan']], y=[row['Minutos']],
        marker=dict(color=PALETA[row['Plan']], line=dict(color='white', width=0)),
        text=f"{row['Minutos']:.0f} min",
        textposition='outside',
        textfont=dict(size=13, color='#374151'),
        name=row['Plan'],
        hovertemplate=f"{row['Plan']}: {row['Minutos']:.0f} min/mes<extra></extra>",
        width=0.45
    ))
fig.update_layout(**base_layout(),
                  xaxis_title='Plan de suscripción',
                  yaxis_title='Minutos promedio / mes',
                  showlegend=False)
fig.update_yaxes(range=[0, medias['Minutos'].max() * 1.2])
st.plotly_chart(fig, use_container_width=True)
st.info("Los usuarios Premium consumen casi el doble que los usuarios Básicos "
        "(≈1.123 vs. ≈587 min/mes). El plan de suscripción es el principal "
        "diferenciador del nivel de consumo en la plataforma.")

st.markdown("---")

# ── BV-2 ─────────────────────────────────────────────────────────────────────
st.subheader("BV-2 — Tickets de soporte vs. tiempo de visualización")

corr = df_f['customer_support_tickets'].corr(df_f['monthly_watch_time_mins'])
avg_by_ticket = (df_f.groupby('customer_support_tickets')['monthly_watch_time_mins']
                 .mean().reset_index())
avg_by_ticket.columns = ['Tickets', 'Promedio']

fig = go.Figure()
fig.add_trace(go.Bar(
    x=avg_by_ticket['Tickets'], y=avg_by_ticket['Promedio'],
    marker=dict(color='#F43F5E', line=dict(color='white', width=0)),
    text=avg_by_ticket['Promedio'].round(0),
    textposition='outside',
    textfont=dict(size=11, color='#374151'),
    hovertemplate='Tickets: %{x}<br>Promedio: %{y:.0f} min/mes<extra></extra>',
    width=0.6
))
fig.add_hline(y=df_f['monthly_watch_time_mins'].mean(),
              line_dash='dash', line_color='#6366F1', line_width=1.5,
              annotation_text=f"Media global: {df_f['monthly_watch_time_mins'].mean():.0f} min",
              annotation_position='top right',
              annotation=dict(font=dict(color='#6366F1', size=11)))
fig.update_layout(**base_layout(),
                  xaxis_title='Cantidad de tickets de soporte',
                  yaxis_title='Minutos promedio / mes',
                  showlegend=False)
fig.update_yaxes(range=[0, avg_by_ticket['Promedio'].max() * 1.2])
fig.add_annotation(x=0.97, y=0.05, xref='paper', yref='paper',
                   text=f'Correlación r = {corr:.3f}',
                   showarrow=False,
                   bgcolor='#FEF3C7', bordercolor='#F59E0B',
                   borderwidth=1, borderpad=6,
                   font=dict(size=12, color='#92400E'))
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
mv1.columns = ['País', 'Plan', 'Minutos']

fig = go.Figure()
for plan in orden_planes:
    datos = mv1[mv1['Plan'] == plan]
    fig.add_trace(go.Bar(
        x=datos['País'], y=datos['Minutos'],
        name=plan,
        marker=dict(color=PALETA[plan], line=dict(color='white', width=0)),
        hovertemplate=f'{plan}: %{{y:.0f}} min/mes<extra></extra>',
    ))
fig.update_layout(**base_layout(height=480),
                  barmode='group',
                  xaxis_title='País',
                  yaxis_title='Minutos promedio / mes',
                  showlegend=True,
                  legend=dict(title='Plan', orientation='h',
                              yanchor='bottom', y=1.02,
                              xanchor='right', x=1,
                              font=dict(size=12)))
fig.update_yaxes(showgrid=True, gridcolor='#F3F4F6')
st.plotly_chart(fig, use_container_width=True)
st.info("El patrón Premium > Estándar > Básico se mantiene consistente en todos los países, "
        "lo que indica que la relación entre plan y consumo no varía por mercado geográfico.")