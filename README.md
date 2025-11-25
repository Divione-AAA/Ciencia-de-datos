# 📚 Ciencia-de-datos

¡Bienvenido/a al repositorio `Ciencia-de-datos`! Este repositorio agrupa proyectos, notebooks y scripts orientados a ciencia de datos, aprendizaje automático y visualización. Aquí encontrarás ejemplos prácticos, experimentos y ejercicios organizados por tema.

**Contenido general**
- **Proyectos principales**: carpetas como `Kmeans Clustering`, `KNN`, `Regresion lineal`, `Regresion logistica`, `Sistemas de Recomendacion`, `Models`, `NLP`, `SVM`, entre otras.
- **Scripts raíz**: `demographic_data_analyzer.py`, `mean_var_std.py`, `medical_data_visualizer.py`, `sea_level_predictor.py`, `time_series_visualizer.py`, etc.
- **Notebooks**: múltiples notebooks `.ipynb` con experimentos, tutoriales y ejercicios.

---

**🔎 Estructura del repositorio**
- `Arboles de decision y bosques aleatorios/` : notebooks y ejemplos sobre árboles de decisión y Random Forest.
- `Conferencias/` : materiales, notebooks y notas de conferencias y charlas (e.g., Albumentations, técnicas de aumento de datos).
- `Kmeans Clustering/` : notebooks con implementación de K-means, análisis de clusters y ejemplos prácticos.
- `KNN/` : proyectos y notebooks sobre K-Nearest Neighbors (clasificación, métricas de distancia).
- `Models/Convulsional/` : modelos CNN (ej. `best_malaria_cnn.h5`) y experiments con TensorFlow/Keras.
- `Regresion lineal/` : ejercicios y datasets para regresión lineal (incluye ejemplos de costo, gradient descent).
- `Regresion logistica/` : notebooks sobre regresión logística y problemas como Titanic, cáncer.
- `Sistemas de Recomendacion/` : notebooks y datasets para recomendaciones (filtrado colaborativo, características básicas).
- `SVM/` : ejemplos y notebooks trabajando Support Vector Machines.
- `Tensores y Variables/` : cuadernos sobre tensores, indexing y operaciones en TensorFlow.

---

**🧩 Detalle por proyecto / carpeta (resumen de lo aprendido)**

- `Kmeans Clustering` 🟣
  - Algoritmos: K-Means, métricas de inercia, método del codo.
  - Habilidades: preprocesamiento, selección del número de clusters, visualización de clusters.

- `KNN` 🔵
  - Algoritmos: K-Nearest Neighbors (clasificación y regresión), métricas (Euclidiana, Manhattan).
  - Habilidades: normalización/standardización, validación cruzada, selección de k.

- `Arboles de decision y bosques aleatorios` 🌲
  - Algoritmos: Decision Trees, Random Forests.
  - Habilidades: importancia de features, overfitting vs. pruning, ensemble learning.

- `Regresion lineal` ➖
  - Algoritmos: regresión lineal simple y múltiple, descenso por gradiente, MSE.
  - Habilidades: análisis de errores, regularización básica (introducida), ingeniería de características.

- `Regresion logistica` 🔐
  - Algoritmos: regresión logística, funciones de pérdida (log-loss), métricas (precision, recall, F1, ROC-AUC).
  - Habilidades: ingeniería de variables categóricas, manejo de datos desbalanceados, evaluación de clasificadores.

- `Models/Convulsional` 🧠
  - Algoritmos: Redes neuronales convolucionales (CNN) con Keras/TensorFlow.
  - Habilidades: diseño de arquitecturas CNN, entrenamiento, checkpoints (`.h5`), uso de callbacks y tensorboard (runs/).

- `Time series / sea_level_predictor / time_series_visualizer` 📈
  - Algoritmos: análisis de series temporales, regresión sobre tiempo, visualización de tendencia y estacionalidad.
  - Habilidades: manipulación de fechas, resampling, smoothing y representación gráfica de series.

- `NLP` 🗣️
  - Algoritmos/Conceptos: tokenización, representación básica de texto y preprocesado para modelos sencillos.
  - Habilidades: limpieza de texto, exploración de datos textuales.

- `SVM` ⚫
  - Algoritmos: Support Vector Machines (márgenes, kernels lineal y no lineal).
  - Habilidades: elección de kernel, ajuste de C y parámetros de regularización.

- `Sistemas de Recomendacion` ⭐
  - Algoritmos: técnicas básicas de recomendación (filtrado colaborativo y contenido básico).
  - Habilidades: manejo de matrices usuario-item, métricas de evaluación (MAE, RMSE, precisión@k).


**🛠️ Herramientas, librerías y habilidades técnicas**
- **Lenguajes:** `Python` (principalmente).
- **Librerías:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `tensorflow`/`keras`.
- **Conceptos ML:** EDA (exploratory data analysis), preprocesamiento, feature engineering, cross-validation, métricas de evaluación, selección de modelos.
- **Deep Learning:** conceptos básicos de CNN, entrenamiento, callbacks y guardado de modelos.
- **Visualización:** gráficos con `matplotlib`, `seaborn` y `plotly` para explorar datos y resultados.
- **Notebooks:** uso intensivo de Jupyter Notebooks para experimentación y visualización de pasos.

---

**📌 Cómo usar este repositorio**
- Abrir los notebooks con Jupyter / VS Code: `jupyter notebook` o abrir directamente los `.ipynb`.
- Para ejecutar scripts sueltos (ejemplos):

```
python demographic_data_analyzer.py
python sea_level_predictor.py
```

- Revisa las carpetas de cada tema para ver datasets (`train.csv`, `House_Price.csv`, etc.) y notebooks con explicaciones paso a paso.

---

**🧭 Buenas prácticas y recomendaciones**
- Empieza por los notebooks de `Inicio/` para familiarizarte con los datos y herramientas.
- Reproduce los experimentos en pequeños pasos: EDA → Preprocesado → Modelado → Evaluación.
- Documenta cambios y resultados en un notebook nuevo para cada experimento.

---

**✉️ Contacto / Autor**
- Repositorio personal de aprendizaje y experimentación: perfecto para compartir avances y material educativo.

---

¡Gracias por explorar `Ciencia-de-datos`! 🚀 Si quieres que genere una versión en inglés, añadir un badge o un índice auto-generado, dímelo y lo hago.
