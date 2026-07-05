import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from pathlib import Path

st.title("🔬 Reducción de Dimensionalidad — PCA")
st.caption("Análisis de componentes principales sobre las variables numéricas del dataset.")

@st.cache_data
def load_and_pca():
    base = Path(__file__).resolve().parents[2]
    df = pd.read_csv(base / 'data/processed/streaming_users_clean.csv',
                     parse_dates=['last_login_date'])
    features = ['age', 'monthly_watch_time_mins', 'customer_support_tickets']
    X = df[features].copy()
    X_scaled = StandardScaler().fit_transform(X)
    X_imp = SimpleImputer(strategy='mean').fit_transform(X_scaled)
    pca = PCA()
    X_pca = pca.fit_transform(X_imp)
    loadings = pd.DataFrame(pca.components_.T, index=features,
                            columns=[f'PC{i+1}' for i in range(len(features))])
    return df, X, X_scaled, X_pca, pca, loadings, features

df, X, X_scaled, X_pca, pca, loadings, features = load_and_pca()
varianza = pca.explained_variance_ratio_
varianza_acum = np.cumsum(varianza)

LAYOUT = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Inter, sans-serif', size=12, color='#374151'),
    margin=dict(t=50, b=50, l=50, r=30),
    hoverlabel=dict(bgcolor='white', bordercolor='#E5E7EB',
                    font=dict(size=12, color='#374151')),
)

st.markdown("---")

# ── VARIABLES Y ESCALAMIENTO ─────────────────────────────────────────────────
st.subheader("Variables y escalamiento")
st.write("Se aplica PCA sobre las tres variables numéricas del dataset, "
         "previamente estandarizadas con **StandardScaler** (media = 0, desvío = 1).")

col1, col2, col3 = st.columns(3)
col1.metric("Variables analizadas", "3")
col2.metric("Registros", f"{len(X):,}")
col3.metric("Escalamiento", "StandardScaler")

col1, col2 = st.columns(2)
with col1:
    st.caption("Antes del escalamiento")
    st.dataframe(pd.DataFrame(X).describe().round(2), use_container_width=True)
with col2:
    st.caption("Después del escalamiento")
    st.dataframe(pd.DataFrame(X_scaled, columns=features).describe().round(3),
                 use_container_width=True)
st.info("Sin escalamiento, `monthly_watch_time_mins` (rango ~0–2.692) dominaría la varianza "
        "del PCA, minimizando la contribución de `age` y `customer_support_tickets`.")

st.markdown("---")

# ── VARIANZA EXPLICADA ────────────────────────────────────────────────────────
st.subheader("Varianza explicada por componente")

col1, col2, col3 = st.columns(3)
col1.metric("PC1", f"{varianza[0]*100:.1f}%")
col2.metric("PC2", f"{varianza[1]*100:.1f}%")
col3.metric("PC3", f"{varianza[2]*100:.1f}%")

# Scree plot
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['PC1', 'PC2', 'PC3'],
    y=varianza * 100,
    marker=dict(color=['#6366F1', '#8B5CF6', '#A78BFA'],
                line=dict(color='white', width=0)),
    text=[f'{v*100:.1f}%' for v in varianza],
    textposition='outside',
    textfont=dict(size=12, color='#374151'),
    name='Varianza por componente',
    hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
    width=0.45
))
fig.add_trace(go.Scatter(
    x=['PC1', 'PC2', 'PC3'],
    y=varianza_acum * 100,
    mode='lines+markers',
    line=dict(color='#F59E0B', width=2.5, dash='dot'),
    marker=dict(size=8, color='#F59E0B', line=dict(color='white', width=2)),
    name='Varianza acumulada',
    hovertemplate='Acumulada: %{y:.1f}%<extra></extra>'
))
fig.add_hline(y=80, line_dash='dash', line_color='#EF4444', line_width=1.5,
              annotation_text='Umbral 80%',
              annotation_position='top right',
              annotation=dict(font=dict(color='#EF4444', size=11)))
fig.update_layout(**LAYOUT,
                  height=420,
                  xaxis=dict(showgrid=False, linecolor='#E5E7EB'),
                  yaxis=dict(showgrid=True, gridcolor='#F3F4F6',
                             linecolor='#E5E7EB', title='Varianza explicada (%)'),
                  legend=dict(orientation='h', yanchor='bottom', y=1.02,
                              xanchor='right', x=1))
st.plotly_chart(fig, use_container_width=True)
st.info(f"La varianza se distribuye de forma casi uniforme: PC1 ≈{varianza[0]*100:.1f}%, "
        f"PC2 ≈{varianza[1]*100:.1f}%, PC3 ≈{varianza[2]*100:.1f}%. "
        "Se necesitan las 3 componentes para superar el 80% acumulado, lo que confirma "
        "que las variables son estadísticamente independientes entre sí.")

st.markdown("---")

# ── BIPLOT ────────────────────────────────────────────────────────────────────
st.subheader("Biplot — PC1 vs PC2")

nombres = ['age', 'monthly_watch_time_mins', 'customer_support_tickets']
etiquetas = ['Edad', 'Consumo mensual', 'Tickets soporte']
colores_vec = ['#EF4444', '#F59E0B', '#10B981']

fig = go.Figure()

# Puntos usuarios
fig.add_trace(go.Scatter(
    x=X_pca[:, 0], y=X_pca[:, 1],
    mode='markers',
    marker=dict(size=4, color='#6366F1', opacity=0.2,
                line=dict(color='white', width=0)),
    name='Usuarios',
    hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
))

# Vectores de carga
escala = 3
for i, (feat, etiq, col) in enumerate(zip(nombres, etiquetas, colores_vec)):
    x_end = pca.components_[0, i] * escala
    y_end = pca.components_[1, i] * escala
    fig.add_annotation(
        x=x_end, y=y_end, ax=0, ay=0,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowsize=1.5,
        arrowwidth=2, arrowcolor=col
    )
    fig.add_trace(go.Scatter(
        x=[x_end * 1.15], y=[y_end * 1.15],
        mode='text',
        text=[etiq],
        textfont=dict(size=12, color=col),
        showlegend=False
    ))

fig.add_hline(y=0, line_color='#E5E7EB', line_width=1)
fig.add_vline(x=0, line_color='#E5E7EB', line_width=1)
fig.update_layout(**LAYOUT,
                  height=480,
                  xaxis=dict(showgrid=True, gridcolor='#F3F4F6',
                             linecolor='#E5E7EB',
                             title=f'PC1 ({varianza[0]*100:.1f}%)'),
                  yaxis=dict(showgrid=True, gridcolor='#F3F4F6',
                             linecolor='#E5E7EB',
                             title=f'PC2 ({varianza[1]*100:.1f}%)'),
                  showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── LOADINGS ──────────────────────────────────────────────────────────────────
st.subheader("Cargas (loadings) por componente")
st.dataframe(loadings.round(4), use_container_width=True)

st.info("**PC1 — Perfil de usuario activo general:** las tres variables contribuyen "
        "de forma similar (age: 0.646, watch_time: 0.554, tickets: 0.525). "
        "Usuarios con valores altos tienen mayor edad, mayor consumo y más tickets.\n\n"
        "**PC2 — Tickets vs. consumo:** contrapone tickets de soporte (0.749) "
        "y tiempo de visualización (-0.661). Diferencia usuarios con muchos tickets "
        "y bajo consumo de usuarios con alto consumo y pocos tickets.\n\n"
        "**PC3 — Edad vs. actividad:** dominada por age (0.762). Diferencia usuarios "
        "mayores con bajo consumo de usuarios jóvenes con alto consumo y más tickets.")
st.markdown("---")
st.subheader("Conclusión")
st.success("El PCA confirma que las tres variables miden dimensiones independientes del usuario. "
           "No hay reducción dimensional útil (se necesitan las 3 componentes), pero el análisis "
           "aporta valor interpretativo: cada componente tiene un significado conceptual claro.")