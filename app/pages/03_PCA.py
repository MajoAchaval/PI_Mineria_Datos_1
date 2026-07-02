import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("🔬 Escalamiento y Reducción de Dimensionalidad — PCA")

@st.cache_data
def load_and_pca():
    df = pd.read_csv('data/processed/streaming_users_clean.csv', parse_dates=['last_login_date'])
    features = ['age', 'monthly_watch_time_mins', 'customer_support_tickets']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(pca.components_.T, index=features,
                            columns=[f'PC{i+1}' for i in range(len(features))])
    return df, X, X_scaled, X_pca, pca, loadings, features

df, X, X_scaled, X_pca, pca, loadings, features = load_and_pca()

st.subheader("Variables utilizadas")
st.write("Se aplica PCA sobre las tres variables numéricas del dataset: `age`, `monthly_watch_time_mins` y `customer_support_tickets`.")
col1, col2, col3 = st.columns(3)
col1.metric("Variables", str(len(features)))
col2.metric("Registros", f"{len(X):,}")
col3.metric("Escalamiento", "StandardScaler")

st.markdown("---")
st.subheader("Justificación del escalamiento")
col1, col2 = st.columns(2)

with col1:
    st.write("**Antes del escalamiento (unidades originales):**")
    st.dataframe(pd.DataFrame(X).describe().round(2), use_container_width=True)
with col2:
    st.write("**Después del escalamiento (media=0, std=1):**")
    st.dataframe(pd.DataFrame(X_scaled, columns=features).describe().round(3), use_container_width=True)

st.info("**Decisión:** Se utiliza StandardScaler porque las variables operan en escalas muy diferentes. "
        "Sin escalamiento, `monthly_watch_time_mins` (rango ~0-5000) dominaría la varianza del PCA, "
        "minimizando la contribución de `age` y `customer_support_tickets`.")

st.markdown("---")
st.subheader("Varianza explicada por componente")

varianza = pca.explained_variance_ratio_
varianza_acum = np.cumsum(varianza)

col1, col2, col3 = st.columns(3)
col1.metric("PC1 explica", f"{varianza[0]*100:.1f}%")
col2.metric("PC2 explica", f"{varianza[1]*100:.1f}%")
col3.metric("PC1+PC2 acumulado", f"{varianza_acum[1]*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, len(varianza)+1), varianza*100, color='steelblue', alpha=0.8)
axes[0].plot(range(1, len(varianza)+1), varianza_acum*100, 'ro-', label='Acumulada')
axes[0].axhline(80, color='orange', linestyle='--', alpha=0.7)
axes[0].set_title('Scree plot — Varianza explicada')
axes[0].set_xlabel('Componente'); axes[0].set_ylabel('%'); axes[0].legend()

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2, s=10, color='teal')
for i, feat in enumerate(features):
    axes[1].arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3,
                  head_width=0.08, color='red', length_includes_head=True)
    axes[1].text(pca.components_[0, i]*3.3, pca.components_[1, i]*3.3, feat, fontsize=9, color='red')
axes[1].set_title(f'Biplot PC1 ({varianza[0]*100:.1f}%) vs PC2 ({varianza[1]*100:.1f}%)')
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].axhline(0, color='gray', lw=0.5); axes[1].axvline(0, color='gray', lw=0.5)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.subheader("Cargas (loadings) por componente")
st.dataframe(loadings.round(4), use_container_width=True)
st.info("**Interpretación:** Los loadings indican cuánto contribuye cada variable original a cada "
        "componente principal. Valores altos (positivos o negativos) señalan las variables que más "
        "definen cada dimensión latente. Completar con la interpretación sustantiva al ejecutar.")
