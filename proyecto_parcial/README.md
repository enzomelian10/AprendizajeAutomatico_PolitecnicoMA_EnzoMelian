<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Proyecto de parcial - Aprendizaje automatico 

# Modelado Predictivo y Segmentación de la Matriz Minera Argentina

Este proyecto desarrolla e implementa un enfoque mixto de Aprendizaje Automático (No Supervisado y Supervisado) sobre datasets históricos del comercio minero global y nacional. El objetivo es estructurar la matriz de recursos de la Argentina, predecir el flujo monetario de las exportaciones a corto plazo (1 a 2 años) y simular escenarios económicos basados en la variación de parámetros de extracción y precios internacionales.

---

## 📌 Índice
- [Introducción y Contexto](#introducción-y-contexto)
- [Objetivos del Proyecto](#objetivos-del-proyecto)
- [Estructura del Proyecto](#organizacion-del-proyecto)
- [Definición del Problema de ML](#definición-del-problema-de-aprendizaje-automático)
- [Origen y Estructura de los Datasets](#origen-y-estructura-de-los-datasets)
- [Preprocesamiento y Procesamiento de Datos](#etapa-de-procesamiento-de-datos)
- [Etapa de Aprendizaje Automático](#etapa-de-aprendizaje-automático)
  - [Clustering (No Supervisado)](#clustering-de-minerales--aprendizaje-no-supervisado)
  - [Regresión y Simulador (Supervisado)](#regresión-y-simulador-what-if---aprendizaje-supervisado)
- [Conclusiones](#conclusiones-del-proyecto)

---

## Introducción y Contexto

El problema central que aborda este trabajo radica en la dificultad para interpretar de forma integrada la compleja estructura de la matriz minera nacional, donde coexisten minerales metalíferos de alto valor unitario, minerales no metalíferos y rocas de aplicación de gran volumen, y cómo esta diversidad impacta en las proyecciones económicas a mediano plazo.

Tradicionalmente, el análisis de este sector se ha limitado a estadísticas descriptivas lineales que miran el pasado, perdiendo la oportunidad de identificar patrones ocultos de competitividad y simular escenarios futuros. En el contexto geopolítico actual, impulsado por la transición energética global y la alta demanda de elementos estratégicos como el litio y el cobre, la minería adquiere un rol crítico para la economía argentina como fuente de divisas. 

Por lo tanto, se vuelve sumamente relevante diseñar e implementar modelos avanzados de Machine Learning que permitan clasificar de manera automatizada el perfil comercial de los recursos mineros y predecir con precisión el flujo monetario de las exportaciones bajo diferentes escenarios productivos.

---

## Objetivos del Proyecto

### Objetivo General
Desarrollar e implementar un enfoque mixto de aprendizaje automático sobre datasets históricos del comercio minero global para estructurar de manera óptima la matriz de recursos de la Argentina, con el fin de predecir el flujo monetario de las exportaciones para los próximos 1 a 2 años y simular escenarios económicos basados en la variación de parámetros de extracción y precios internacionales.

### Objetivos Específicos
1. **Segmentar y caracterizar la matriz minera:** Aplicar algoritmos de agrupamiento (_clustering_) para clasificar automáticamente los diferentes minerales y países exportadores en perfiles económicos homogéneos basados en su volumen, precio internacional y estabilidad en el mercado.
2. **Proyectar el flujo monetario por sectores:** Entrenar modelos de regresión para predecir las exportaciones futuras de la Argentina a mediano plazo, desglosando las proyecciones según los grupos de minerales identificados (metalíferos, no metalíferos y rocas de aplicación).
3. **Desarrollar un simulador de escenarios ("What-If"):** Diseñar una función predictiva que permita modificar parámetros clave (como el aumento del volumen de extracción de un mineral o la caída de su precio) para evaluar el impacto neto en el ingreso de divisas al país.
4. **Generar un marco de visualización avanzada:** Reducir la alta dimensionalidad de los datos comerciales globales para representar espacialmente el posicionamiento competitivo y el estado actual de la Argentina en el mercado internacional.

---

## Organizacion del proyecto

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         proyecto_parcial and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── proyecto_parcial   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes proyecto_parcial a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---
## Definición del Problema de Aprendizaje Automático

Para abordar el problema planteado de manera integral, el proyecto se divide en dos etapas de naturalezas distintas:

* **Etapa de Aprendizaje No Supervisado (Clustering):** No se busca predecir una etiqueta conocida, sino descubrir la estructura oculta de los datos para clasificar los minerales industriales, metales y rocas en perfiles comerciales basados en características comunes de mercado.
* **Etapa de Aprendizaje Supervisado (Regresión):** Se trata estrictamente de un problema de regresión. La variable objetivo a predecir, el valor FOB de las exportaciones en millones de USD, es una variable continua y numérica a lo largo del tiempo.

---

## Origen y Estructura de los Datasets

El proyecto se nutre de cuatro fuentes de datos principales extraídas de portales oficiales:

- https://datos.gob.ar/dataset/mineria-comercio-mundial-mineria
- https://portal-andino.datos.gob.ar/dataset/mineria-demo-comercio-minerales-argentina-siacam/archivo/mineria-demo_14.3

| # | Dataset | Variables Clave | Registros | Utilidad Técnica | Enlace al dataset |
|---|---|---|---|---|---|
| 1 | **Exportaciones Mineras por País** | `year`, `pais_exportador`, `value` | 6.535 | Ver volumen bruto histórico manejado por cada nación. | [Dataset_1](./data/raw/1-expo_CEPII_BACI_mineras_mundiales_pais.csv) |
| 2 | **Comercio Mundial por Sector** | `year`, `sector`, `grupo`, `value` | 2.666 | Aporta la demanda y peso de cada mineral a nivel mundial (variable contextual). | [Dataset_2](./data/raw/2-comex_CEPII_BACI_mineras_mundiales_sector_grupo.csv) |
| 3 | **Porcentaje de Participación** | `year`, `pais_exportador`, `Porcentaje_Participacion` | 6.535 | Métrica normalizada ideal para evitar sesgos por "tamaño de economía". | [Dataset_3](./data/raw/3-expo_CEPII_BACI_mineras_mundiales_porcentajes_pais.csv) |
| 4 | **Exportaciones de Argentina (SIACAM)** | `ANYO`, `MES`, `GRUPO`, `SECTOR`, `FOB` | Desagregado | Único dataset nativo con datos mensuales y ultra-desglosados de Argentina. | [Dataset_4](./data/raw/4-expominerales-siacam.csv) |

### Consideraciones de Integración
* **Normalización de Moneda:** Los valores mundiales se expresan en millones de USD, mientras que el dataset de SIACAM figura en dólares corrientes (unidades). Se transformó el valor FOB de Argentina a millones de USD para unificar magnitudes.
* **Normalización Temporal:** Se consolidaron los registros mensuales de Argentina sumándolos anualmente por grupo y sector para emparejarlos con la escala anual de los datasets internacionales.
* **Estrategia de Partición Temporal:** Se utilizaron los registros de **1995 hasta 2024 para el entrenamiento** de los modelos. Los registros de **2025 y 2026 se reservaron exclusivamente para validación** y simulación fuera de muestra.

---

## Etapa de Procesamiento de Datos

Durante esta fase crítica se ejecutaron las siguientes tareas de ingeniería de datos y limpieza:
* Conversión de tipos de datos a numéricos (ej. `Year` y `Value`).
* Tratamiento de valores nulos y eliminación de registros duplicados en los 4 conjuntos de datos.
* Estandarización y limpieza de etiquetas en variables categóricas (particularmente `Sector` y `Grupo`).
* Agrupación temporal y separación de los años de validación (2025 y 2026) del dataset SIACAM.

---

## Etapa de Aprendizaje Automático

### Modelos y Métodos Utilizados
* **Estandarización:** `StandardScaler` para normalizar medias a 0 y desvíos estándar a 1, mitigando el impacto de escalas dispares (ej. gramos de oro vs. toneladas de caliza).
* **Reducción de Dimensionalidad:** `PCA` para remover multicolinealidad económica y mapear el posicionamiento en un gráfico 2D.
* **Algoritmos de Clustering:** `K-Means` y `DBSCAN`.
* **Algoritmos de Regresión:** `Random Forest Regressor` y modelos de series temporales (`Prophet`/`ARIMA`) para contrastar métricas.

### Clustering de Minerales – Aprendizaje No Supervisado
A partir del método del Codo, se determinó que la cantidad óptima de agrupaciones para la matriz comercial global es **k = 3**:
* **Cluster 0:** Compuesto por 84 minerales de bajo volumen promedio y presencia estable. Es el clúster menos volátil en el mercado.
* **Clusters 1 y 2:** Grupos reducidos (1 y 2 minerales respectivamente) con alta relevancia estratégica o volúmenes masivos. Presentan valores promedio muy altos pero volatilidades igualmente elevadas.

> 🛠️ **Refinamiento con DBSCAN:** Se ejecutó para aislar de forma orgánica los *outliers* masivos del mercado internacional. El algoritmo identificó con precisión los recursos clave de la economía minera argentina: **Cobre, Hierro, Litio, Oro y Plata**.
> 
> El análisis posterior mediante **PCA** reflejó que, a escala global, Argentina se ubica todavía en un segmento de volumen de exportación bajo, pero demostrando una constancia de crecimiento sostenido a lo largo de la serie temporal.

### Regresión y Simulador "What-If" – Aprendizaje Supervisado
Para modelar el comportamiento del dataset unificado de Argentina, se generaron variables basadas en retrasos temporales (*lags*) y se codificaron los atributos categóricos. 

Se entrenó un **Random Forest Regressor** para predecir el valor FOB en función del sector y la inercia de periodos pasados, alcanzando una métrica de rendimiento sobresaliente:

$$\mathbf{R^2 = 0,929}$$

El análisis de *Feature Importance* confirmó que el valor de exportación del año inmediatamente anterior (`fob_lag_1`) es el predictor más fuerte, evidenciando una marcada inercia sectorial.

#### Validación (Predicción Año 2025)
Al testear el modelo con datos no vistos del año 2025 utilizando el histórico 1994-2024, se obtuvieron las siguientes tasas de error:
* **Minerales Industriales y Rocas de Aplicación:** Errores inferiores al **7%**.
* **Sector Metalífero:** Error cercano al **32%** (atribuible a la alta sensibilidad de este sector a factores exógenos y shocks macroeconómicos internacionales que rompen la inercia temporal estricta).

#### Simulador Predictivo 2026
Se encapsuló una función predictiva final orientada a calcular el valor FOB estimado para el año 2026 combinando los retrasos históricos junto con variaciones porcentuales hipotéticas (parámetros *What-If* de volumen o precio) ingresadas por el usuario.

---

## Conclusiones del proyecto

La combinación de técnicas no supervisadas y supervisadas permitió caracterizar con rigor matemático la estructura de la minería argentina, abstrayendo patrones comerciales clave y aislando los *outliers* de alto valor estratégico. El modelo predictivo final cuenta con una elevada capacidad explicativa ($R^2 = 0,929$), y la implementación del simulador de escenarios proporciona una herramienta sólida para asistir a la toma de decisiones estratégicas, permitiendo mensurar el impacto potencial de políticas productivas o shocks de precios sobre el ingreso de divisas al país.

--------
