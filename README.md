# PI — Minería de Datos 1
## Análisis de Usuarios de Plataforma de Streaming

---

## Información general

Proyecto Integrador de la asignatura Minería de Datos 1.
Comisión: Turno Tarde, Nodo | Integrante: Achával María José

---

## Objetivo del proyecto

Aplicar los contenidos de Minería de Datos 1 para construir un análisis de datos reproducible
sobre un dataset de usuarios de una plataforma de streaming latinoamericana. Se busca explorar
patrones de consumo, identificar problemas de calidad de datos, preparar el dataset con decisiones
justificadas y aplicar reducción de dimensionalidad mediante PCA.

---

## Dataset

El archivo `streaming_users_dirty.json` contiene 8.160 registros de usuarios con 8 variables:
`user_id`, `age`, `subscription_plan`, `monthly_watch_time_mins`, `country`, `favorite_genre`,
`last_login_date` y `customer_support_tickets`. El dataset presenta problemas intencionales de
calidad: valores negativos e imposibles, variantes de texto inconsistentes, fechas futuras y filas
duplicadas. El dataset original se preserva sin modificaciones en `data/raw/`.
Referencia: Cátedra de Minería de Datos 1.

---

## Estructura del repositorio

```
PI_Mineria_Datos_1/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                   # Dataset original sin modificaciones
│   └── processed/             # Dataset final utilizado en el análisis
├── notebooks/
│   ├── 01_inspeccion_inicial.ipynb
│   ├── 02_calidad_y_limpieza.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_pca.ipynb
│   └── 05_conclusiones.ipynb
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 01_Dataset.py
│       ├── 02_EDA.py
│       ├── 03_PCA.py
│       └── 04_Conclusiones.py
├── reports/
│   └── informe_final.pdf
└── logs/
    └── pipeline_log.csv
```

---

## Preparación y calidad de datos

El proceso de limpieza se documenta en `notebooks/02_calidad_y_limpieza.ipynb` y se registra en
`logs/pipeline_log.csv`. Se realizaron 7 pasos principales: eliminación de duplicados exactos y
por `user_id` (160 filas, 8.160 → 8.000), normalización de `subscription_plan` (variantes →
3 categorías canónicas), normalización de `country` (variantes → 7 países) y de `favorite_genre`
(variantes → 7 géneros), conversión a `NaN` de valores imposibles en `age` (<13 o >100, 120
registros), `monthly_watch_time_mins` (negativos o ≥10.000 min, 80 registros) y
`customer_support_tickets` (negativos o valores atípicos 99/150, 96 registros), tratamiento de
outliers en `monthly_watch_time_mins` mediante winsorización (límite superior Q3 + 3·IQR ≈ 2.692
minutos, 108 valores acotados), y conversión/validación de `last_login_date` (320 nulos
originales + fechas futuras posteriores al 2026-06-24 anuladas, dejando 399 valores `NaT`
finales sin imputar). Los valores faltantes en `age` y `monthly_watch_time_mins` se imputaron
con la mediana agrupada por `subscription_plan` (mecanismo diagnosticado como MAR), mientras que
`customer_support_tickets` se imputó con la mediana global. La retención final fue de
**98.0%** del dataset original (8.000 de 8.160 filas).

---

## Resumen del análisis exploratorio

El EDA se desarrolla en `notebooks/03_eda.ipynb` y se presenta en la aplicación Streamlit
(página EDA). Se analizaron 5 visualizaciones: distribución de edades (UV-1) — concentrada
entre 25 y 45 años, media 33.6 / mediana 33.0 —, distribución del tiempo de visualización
(UV-2) — asimétrica a la derecha, media ≈790 min/mes, mediana ≈771 min/mes —, tiempo promedio
por plan de suscripción (BV-1) — Premium (≈1.123 min) > Estándar (≈862 min) > Básico (≈587 min),
confirmando la hipótesis de mayor consumo en planes superiores —, relación entre tickets de
soporte y tiempo de visualización (BV-2) — sin tendencia clara, consumo estable en ≈700-800
min/mes salvo el grupo con 5 tickets (≈1.029 min) —,y tiempo de visualización por país y plan de suscripción (MV-1) — el patrón Premium > Estándar > Básico se mantiene consistente en los 7 países analizados, sin variación significativa por mercado geográfico.

---

## Reducción de dimensionalidad

El PCA se desarrolla en `notebooks/04_pca.ipynb` y se presenta en la página PCA de Streamlit.
Se aplicó StandardScaler sobre las tres variables numéricas continuas (`age`,
`monthly_watch_time_mins`, `customer_support_tickets`) previo al PCA, dado que operan en escalas
muy diferentes. La varianza explicada se distribuye de forma casi uniforme entre las tres
componentes (PC1 ≈33.8%, PC2 ≈33.2%, PC3 ≈33.0%), por lo que se requieren las **3 componentes**
para alcanzar el umbral del 80% de varianza acumulada. Esta distribución uniforme es consistente
con la baja correlación entre variables observada en el EDA: al no existir redundancia, el PCA no
logra una reducción dimensional efectiva, aunque sí aporta valor interpretativo — PC1 combina
edad y consumo (compromiso general), PC2 contrapone edad y consumo (jóvenes de alto consumo vs.
mayores de bajo consumo), y PC3 está dominada por `customer_support_tickets` (interacción con
soporte). Ver `notebooks/04_pca.ipynb`.

---

## Visualización interactiva

Aplicación pública en Streamlit Cloud: https://pi-mineriadatos1.streamlit.app/
El repositorio se referencia en la página de inicio de la aplicación.

---

## Cómo ejecutar localmente

```bash
git clone https://github.com/MajoAchaval/PI_Mineria_Datos_1.git
cd PI_Mineria_Datos_1
pip install -r requirements.txt
# Ejecutar notebooks en orden (01 → 05)
cd app
streamlit run Home.py
```

---

## Conclusiones

El plan de suscripción es la variable más predictiva del consumo: los usuarios Premium ven
casi el doble que los Básicos (≈1.123 vs. ≈587 min/mes), patrón consistente en todos los
países analizados. Las tres variables numéricas (`age`, `monthly_watch_time_mins`,
`customer_support_tickets`) son estadísticamente independientes entre sí (r < 0.01 en todos
los pares), lo cual fue confirmado por el PCA al distribuir la varianza casi uniformemente
entre sus tres componentes (~33% cada una): no hay redundancia de información, por lo que el
PCA aporta valor interpretativo más que de reducción dimensional. Contra lo esperado, los
tickets de soporte no se relacionan con un menor consumo (r ≈ 0.003), por lo que no deben
usarse como proxy aislado de insatisfacción o riesgo de abandono. Como limitación principal,
`last_login_date` retiene 399 valores `NaT` sin imputar y el dataset carece de variables de
churn explícito, lo que impide construir modelos predictivos de abandono directamente con esta
información.

---

Informe final: `reports/informe_final.pdf` | Log ETL: `logs/pipeline_log.csv` | Aplicación: https://pi-mineriadatos1.streamlit.app/
