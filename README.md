<p align="center">
  <img src="image%20Home%20Credit.png" alt="Home Credit Banner" width="100%">
</p>


# 📌 Credit Risk Prediction – Home Credit

Proyecto de **predicción de riesgo de crédito** utilizando datos reales del caso Home Credit Default Risk.  
El **objetivo es estimar la probabilidad de impago** de cada cliente mediante un modelo de clasificación robusto, interpretable y reproducible.

El dataset original es complejo y multitabla, con información financiera, del buró de crédito y del historial de pagos.  
Este proyecto reproduce un flujo real de scoring bancario: desde la integración de datos y el EDA hasta la interpretación con SHAP y la generación de predicciones finales.

Este repositorio forma parte de mi formación como **Data Analyst / Data Scientist**, y demuestra un flujo profesional y completo de modelado de riesgo.

---

## 📁 Estructura del repositorio

Este repositorio contiene el notebook principal del proyecto, junto con los artefactos reproducibles generados en el proceso de modelado:

```text
credit-risk-prediction-home-credit/
│
├── credit-risk-prediction-home-credit.ipynb      # Notebook principal (EDA + ML + SHAP + scoring)
│
├── artifacts/                                    # Artefactos reutilizables
│   ├── preprocessing_pipeline_v3.joblib          # Pipeline de preprocesamiento (fitted)
│   ├── feature_names_v3.npy                      # Nombres de las 357 features finales
│   └── lgbm_optimized_v1.joblib                  # Modelo LightGBM optimizado
│
├── submissions/
│   └── submission_lightgbm_20251210_1343.csv     # Predicciones finales para test
│
└── README.md
```


## 🎯 Objetivo del proyecto

El proyecto aborda el problema desde dos perspectivas complementarias: **negocio** (decisiones de riesgo) y **técnica** (construcción de un modelo fiable y reproducible).

### **Objetivo de negocio (Riesgo de Crédito)**

- Estimar la **probabilidad de impago** de cada solicitante.
- Reducir pérdidas evitando conceder préstamos a clientes de alto riesgo.
- Mantener la aprobación de clientes solventes para no afectar la rentabilidad.
- Utilizar un modelo **interpretativo y justificable**, adecuado para auditorías y entornos regulados.

### **Objetivo técnico**

- Construir un **pipeline reproducible** de preparación, modelado y scoring.
- Evaluar diferentes algoritmos y seleccionar el modelo óptimo.
- Emplear **métricas específicas del sector financiero** (AUC, KS) para garantizar validez en riesgo de crédito.
- Incorporar **interpretabilidad con SHAP** para justificar predicciones individuales y globales.
- Generar un archivo final de scoring listo para integrarse en sistemas de decisión.


## 📊 Datos

El proyecto utiliza múltiples tablas del dataset **Home Credit Default Risk**, que se integran mediante un proceso de *feature engineering* para generar un dataset enriquecido adecuado para el modelado.

Tras esta integración se obtiene:

- **train:** 307 511 filas, 524 columnas  
- **test:** 48 744 filas, 523 columnas  

La diferencia de columnas entre train y test se debe a categorías presentes solo en el entrenamiento, lo que refuerza la necesidad de un preprocesamiento robusto con `OneHotEncoder(handle_unknown="ignore")`.

El dataset enriquecido se guarda como un *checkpoint* intermedio, lo que permite:

- acelerar el ciclo de experimentación,  
- garantizar consistencia en los análisis,  
- evitar repetir procesos costosos de ingeniería de características.

Este checkpoint es la base sobre la que se realiza el EDA y se entrena el modelo final.


## 🔍 EDA y selección de variables

El análisis exploratorio permitió evaluar la calidad del dataset enriquecido y detectar aspectos críticos para la construcción de un modelo de riesgo de crédito. Entre los principales hallazgos se encuentran:

- **Distribuciones sesgadas:** variables financieras como ingresos, importes o duración del crédito mostraron fuerte asimetría y presencia de outliers, lo que motivó el uso de imputación robusta (mediana) y controles específicos.
- **Valores faltantes:** varias columnas presentaban porcentajes elevados de nulos; se eliminaron aquellas con ausencia excesiva y se definió una estrategia de imputación diferenciada para variables numéricas y categóricas.
- **Outliers relevantes:** se identificaron valores extremos que podían distorsionar el entrenamiento, especialmente en ingresos y montos, justificando su tratamiento.
- **Correlaciones altas:** se detectaron grupos de variables muy correlacionadas (especialmente en información de buró), lo que llevó a reducir redundancia para mejorar la estabilidad del modelo.
- **Cardinalidad excesiva en categóricas:** algunas variables tenían demasiados niveles, lo que habría generado cientos de columnas tras la codificación. Se filtraron para evitar sobrecarga y ruido.
- **Desbalance de clases:** TARGET=1 era significativamente menos frecuente, reforzando la necesidad de usar métricas adecuadas como AUC y KS.

Tras este proceso se definió un conjunto final de **357 variables limpias, estables y modelables**, que sirvió de base para el pipeline de preprocesamiento y el entrenamiento del modelo.


## ⚙️ Preprocesamiento (pipeline v3)

El objetivo del preprocesamiento es transformar el dataset enriquecido en un conjunto **limpio, consistente y totalmente numérico**, garantizando que tanto *train* como *test* reciben exactamente las mismas transformaciones.

El diseño del pipeline v3 se basa directamente en los hallazgos del EDA.

---

### **1. Variables numéricas**
- **Imputación con mediana:** seleccionada por su robustez frente a outliers, frecuentes en variables financieras como ingresos o importes.  
- **Sin escalado:** modelos basados en árboles (como LightGBM) no requieren normalización ni estandarización, lo que evita transformaciones innecesarias.

---

### **2. Variables categóricas**
- **Imputación con moda:** adecuada para categorías con distribución muy concentrada, evitando crear niveles artificiales.  
- **Codificación con OneHotEncoder (`handle_unknown="ignore"`):**  
  - evita errores cuando aparecen categorías nuevas en el conjunto de test,  
  - mantiene estabilidad en scoring,  
  - es la técnica más adecuada cuando las categóricas no tienen cardinalidad excesiva (las de alta cardinalidad fueron filtradas en el EDA).

---

### **3. Salida del pipeline**
Tras la imputación y codificación, el dataset final se transforma en una matriz de **357 features estables**, lista para el entrenamiento del modelo.

El pipeline se guarda como artefacto reproducible en:

artifacts/preprocessing_pipeline_v3.joblib


Este diseño asegura que **no existe fuga de información** y que el proceso de scoring es **100% consistente y replicable** con respecto al entrenamiento.



## 🤖 Modelado

Se evaluaron varios algoritmos de clasificación con el objetivo de identificar un modelo capaz de discriminar de forma robusta entre clientes con alta y baja probabilidad de impago.  
La comparación se realizó utilizando un conjunto de validación estratificado y métricas estándar del sector financiero (AUC y KS).

### **Modelos comparados**
- Logistic Regression  
- CatBoost  
- XGBoost  
- **LightGBM (modelo final seleccionado)**  

---

### **Resultados en validación**

| Modelo               | AUC      | KS      |
|---------------------|----------|---------|
| **LightGBM**        | **0.7783** | **0.4245** |
| XGBoost             | 0.7771   | 0.4246 |
| CatBoost            | 0.7733   | 0.4175 |
| Logistic Regression | 0.6436   | 0.2135 |

---

### 🎯 **Conclusiones del modelado**

El análisis comparativo muestra que **LightGBM ofrece el mejor equilibrio entre rendimiento, estabilidad y velocidad**, con métricas alineadas con los estándares de modelos productivos en scoring de crédito:

- **AUC ≈ 0.78** → buena capacidad discriminante en entornos con fuerte desbalance de clases.  
- **KS ≈ 0.42** → nivel propio de modelos sólidos en banca minorista.

Además, LightGBM resulta especialmente adecuado para este problema debido a:

- su **robustez frente a valores faltantes y outliers**, frecuentes en datos financieros reales,  
- su capacidad para manejar **miles de variables heterogéneas**,  
- su excelente relación **velocidad / rendimiento** en comparación con otros modelos de boosting,  
- la facilidad para integrarse con técnicas de interpretabilidad como **SHAP**.

Por estos motivos, **LightGBM fue seleccionado como el modelo final** para el despliegue del sistema de scoring.


## 📈 Interpretación de métricas (contexto de riesgo de crédito)

La evaluación del modelo se centra en dos métricas fundamentales en scoring financiero: **AUC** y **KS**.  
Ambas permiten medir la capacidad del modelo para separar buenos y malos pagadores en entornos con fuerte desbalance de clases.

---

### **AUC (Area Under the ROC Curve)**  
Mide la capacidad del modelo para distinguir correctamente entre clientes solventes e insolventes.  

Un AUC de **≈ 0.78** indica un modelo:

- **sólido y estable**,  
- adecuado para problemas reales de riesgo de crédito,  
- robusto ante desbalance de clases (TARGET = 1 es minoritario).

---

### **KS (Kolmogorov–Smirnov)**  
Es la métrica más utilizada en banca, ya que refleja la **separación real entre las distribuciones de buenos y malos clientes**.

Referencias del sector:
- 0.20 → aceptable  
- 0.30 → bueno  
- **0.40+ → muy bueno**

El modelo final obtiene **KS ≈ 0.42**, lo que implica:

- **excelente poder discriminante**,  
- alta capacidad para reducir pérdidas por impago,  
- comportamiento consistente y reproducible en validación.

---

El modelo final **LightGBM** se almacena como artefacto reproducible en:

`artifacts/lgbm_optimized_v1.joblib`


## 🧪 Optimización del modelo (LightGBM)

Para mejorar el rendimiento del modelo base, se realizó una búsqueda de hiperparámetros mediante **RandomizedSearchCV**, una técnica eficiente para explorar espacios amplios sin el elevado coste computacional de Grid Search.

### **Hiperparámetros optimizados**

Los parámetros ajustados fueron:

- `n_estimators`  
- `learning_rate`  
- `num_leaves`  
- `max_depth`  
- `min_child_samples`  
- `subsample`  
- `colsample_bytree`  
- `reg_alpha`  
- `reg_lambda`  

Estos parámetros son especialmente críticos en riesgo de crédito porque controlan:

- la complejidad del modelo,  
- el riesgo de sobreajuste,  
- la capacidad de generalización.

### **Mejores resultados obtenidos**

- **AUC Train:** 0.8255  
- **AUC Valid:** 0.7798  
- **KS Valid:** 0.4259  

La diferencia entre train y valid es **moderada**, lo que indica:

- buena capacidad de generalización,  
- ausencia de sobreajuste significativo,  
- estabilidad del modelo para scoring real.

El modelo final se guarda como artefacto reproducible en:
artifacts/lgbm_optimized_v1.joblib


## 🧠 Interpretabilidad con SHAP

En riesgo de crédito no basta con obtener buenas métricas: es imprescindible **explicar** por qué un modelo asigna a un cliente una probabilidad alta o baja de impago.  
Por este motivo se aplicó **SHAP** para analizar la contribución de cada variable al modelo LightGBM.

### 🔍 ¿Qué aporta SHAP en este proyecto?

Con SHAP pudimos:

- identificar qué variables **incrementan o reducen** la probabilidad de impago,
- comprender patrones globales del riesgo en el portafolio,
- justificar decisiones de crédito ante negocio y auditoría,
- detectar comportamientos no intuitivos o relaciones no lineales capturadas por LightGBM.

### 📌 Principales hallazgos (extraídos del análisis SHAP)

Los resultados de SHAP confirmaron patrones esperados en modelos de scoring:

- Los **retrasos previos en pagos** y variables relacionadas con morosidad fueron las que **más aumentaron la probabilidad de impago**.
- La información del **buró crediticio** (créditos activos, atrasos históricos, montos pendientes) mostró una influencia significativa.
- El **ratio entre pagos e ingresos** y otros indicadores de capacidad de pago tuvieron un impacto importante.
- Una mayor **intensidad del historial crediticio** (número y antigüedad de productos) tendió a reducir el riesgo al aportar estabilidad.

Estas conclusiones están alineadas con lo que se observa en modelos reales de bancos y entidades financieras.

### 🎯 Valor para el negocio

La interpretabilidad aportada por SHAP garantiza:

- **transparencia** del modelo,
- **trazabilidad** de cada decisión,
- cumplimiento de requisitos regulatorios,
- confianza para su potencial uso en un sistema de aprobación de crédito.


## 📤 Scoring final

La fase de scoring aplica el **pipeline de preprocesamiento** y el **modelo final optimizado** para generar la probabilidad de impago (PD) de nuevos clientes utilizando el conjunto de test.

### 🔄 Proceso de scoring
1. Se carga el dataset enriquecido de test.
2. Se aplica el pipeline `preprocessing_pipeline_v3.joblib` para asegurar exactamente las mismas transformaciones que en entrenamiento.
3. El modelo `lgbm_optimized_v1.joblib` genera la probabilidad estimada de impago para cada cliente.
4. Se construye el archivo final de predicciones:
   submissions/submission_lightgbm_20251210_1343.csv


### 📄 Contenido del archivo de salida
- `SK_ID_CURR` — identificador único del cliente.  
- `TARGET` — **probabilidad estimada de impago (PD)** generada por el modelo.

### 🧾 Utilidad del scoring
Este fichero constituye la salida estándar de un sistema de riesgo de crédito y puede integrarse directamente en:
- motores de decisión automáticos,
- validaciones internas,
- simulaciones de políticas de crédito,
- análisis posteriores de negocio o regulador.

El proceso garantiza **consistencia, replicabilidad y ausencia de fugas de información**, ya que train y test pasan por el mismo pipeline.


## 🔁 Reproducibilidad

El proyecto está diseñado para ser **totalmente reproducible**, de forma que cualquier usuario pueda replicar el entrenamiento, el preprocesamiento y el scoring sin modificar el código original.

Los elementos clave que garantizan esta reproducibilidad son:

- Todo el flujo está concentrado en el notebook `credit-risk-prediction-home-credit.ipynb`.
- El preprocesamiento y el modelo se encapsulan en artefactos (`joblib`, `npy`) dentro de la carpeta `artifacts/`.
- El mismo pipeline que se usa para entrenar se usa para hacer scoring, evitando inconsistencias entre train y test.

Para reproducir el proyecto, los pasos generales son:

1. Clonar el repositorio:
```bash
git clone https://github.com/alejandroalvarezselva/credit-risk-prediction-home-credit.git
```
2. Descargar los datos originales desde Kaggle (Home Credit Default Risk).
3. Ejecutar el notebook en Google Colab o entorno local, ajustando las rutas según sea necesario.

Con estos pasos se puede replicar todo el flujo: EDA, preprocesamiento, entrenamiento, evaluación, interpretabilidad y scoring final.


## 📝 Conclusiones

El proyecto permitió construir un sistema completo y reproducible de **predicción de riesgo de crédito**, siguiendo todas las fases del ciclo de modelado.  
Las conclusiones clave por etapa son:

### 🔍 1. EDA y selección de variables
- Se identificó un **fuerte desbalance de clases**, lo que justificó el uso de métricas como AUC y KS en lugar de accuracy.  
- Varias variables financieras mostraron **alta dispersión y outliers**, lo que llevó a utilizar imputación robusta mediante mediana.  
- Se detectaron **correlaciones elevadas** entre indicadores de historial crediticio, reduciendo variables redundantes para evitar inflación de información.  
- Algunas categóricas tenían **cardinalidad excesiva**, lo que confirmó la necesidad de filtrado previo y `OneHotEncoder(handle_unknown="ignore")`.

### ⚙️ 2. Preprocesamiento
- El pipeline v3 permitió transformar el dataset enriquecido en una matriz estable de **357 features**, aplicando exactamente las mismas transformaciones en train y test.  
- Este diseño eliminó riesgos de **fuga de información** y garantizó consistencia total en el scoring final.

### 🤖 3. Modelado y validación
- LightGBM fue el modelo seleccionado tras comparar varias alternativas (Logistic Regression, XGBoost, CatBoost).  
- Las métricas obtenidas (**AUC ≈ 0.78**, **KS ≈ 0.42**) demuestran un **modelo sólido de scoring bancario**, con buena separación entre clientes de alto y bajo riesgo.  
- La diferencia moderada entre train y valid confirma **ausencia de sobreajuste** y buena capacidad de generalización.

### 🧠 4. Interpretabilidad
- SHAP permitió identificar los factores más relevantes del riesgo, destacando:
  - historial de morosidad,  
  - variables del buró,  
  - ratio de deuda/ingresos.  
- Esto aporta **transparencia y trazabilidad**, esenciales en entornos regulados.

### 📦 5. Implementación y reproducibilidad
- El uso de artefactos (`pipeline`, `feature_names`, `modelo`) hace que el proyecto sea **100% replicable**.  
- El archivo de scoring final puede integrarse directamente en un sistema de decisión de crédito.

En conjunto, se obtuvo un **modelo robusto, interpretable y listo para integración**, demostrando un flujo profesional completo de Data Science aplicado al riesgo de crédito.


## 👤 Sobre mí

Soy un profesional en formación con enfoque en **Data Analytics** y en transición hacia **Data Science**, interesado en aplicar el análisis de datos y el machine learning para resolver problemas reales de negocio.

Durante mi aprendizaje he trabajado especialmente con:

- **Python** para análisis, modelado y visualización.  
- Técnicas de **preprocesamiento y preparación de datos**.  
- Modelos supervisados aplicados a casos reales (como el **scoring de crédito**).  
- **Interpretabilidad de modelos** mediante SHAP y análisis de variables.  
- Métricas orientadas a negocio y validación de modelos.

Me motiva construir soluciones basadas en datos que aporten valor, combinen rigor analítico y sean aplicables en entornos reales.

Actualmente busco **mi primera oportunidad profesional** como **Data Analyst** o **Data Scientist Junior**, y estoy abierto a colaborar en proyectos donde pueda seguir aprendiendo y aportando valor.

