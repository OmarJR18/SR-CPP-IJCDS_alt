# Observaciones que requieren implementacion de codigo

Ordenadas de menor a mayor complejidad/tiempo estimado.

---

## 1. Obs 22 — Comparacion de eficiencia computacional

**Observacion:** "Add a computational efficiency comparison showing training time, pruning time, inference time, and memory usage for all pruning methods."

**Solucion:** Medir y reportar para cada metodo:
- `training_time`: tiempo de `fit()`
- `pruning_time`: tiempo del proceso de poda
- `inference_time`: tiempo de `predict()` (ya se mide en `evaluate()`)
- `memory_usage`: `tracemalloc` del modelo serializado

**Implementacion:**
- Modificar funciones `train_*` para devolver `(model, timing_dict)`
- Modificar `evaluate()` para incluir memoria
- Agregar columnas al DataFrame de resultados

**Dependencias nuevas:** Ninguna
**Dificultad:** Baja
**Tiempo estimado:** ~30 min

---

## 2. Obs 11 — Analisis de significancia estadistica

**Observacion:** "The results section should include statistical significance analysis (e.g., paired t-test, Wilcoxon signed-rank test, or Friedman/Nemenyi tests) to demonstrate that the reported improvements are statistically significant."

**Solucion:** Ya existe ANOVA en el notebook. Se extendio el analisis para cubrir tambien las metricas estructurales del arbol:
- **Friedman test** (non-parametric, comparacion de multiples metodos)
- **Nemenyi test** (post-hoc, que pares de metodos difieren significativamente)
- **Wilcoxon signed-rank test** (par por par, SR-CCP/CCP_Modified vs cada metodo)
- **Paired t-test** con correccion de Benjamini-Hochberg (FDR)

**Implementacion:**
- Usar `scipy.stats.friedmanchisquare`
- Generar tablas de p-values para accuracy y para `Leaves`, `Nodes`, `Depth` y `n_Features`
- Guardar el resumen estructural en `FINAL_RESULTADOS_structural_significance.csv`

**Dependencias nuevas:** Ninguna (scipy ya instalado)
**Dificultad:** Baja-Media
**Tiempo estimado:** ~45 min

---

## 3. Obs 14 — Estudio de ablacion

**Observacion:** "An ablation study should be included to evaluate the independent contribution of the depth penalty and feature penalty components within the proposed SR-CCP objective function."

**Solucion:** Evaluar SR-CCP en 4 configuraciones:
1. SR-CCP completo (alpha + depth + feature)
2. SR-CCP sin depth penalty (alpha + feature)
3. SR-CCP sin feature penalty (alpha + depth)
4. SR-CCP solo alpha (como CCP standard)

**Implementacion:**
- Crear funcion `train_ccp_ablation(X_train, y_train, X_val, y_val, config)`
- Ejecutar cross-validation para cada configuracion
- Comparar metricas entre configuraciones

**Dependencias nuevas:** Ninguna
**Dificultad:** Media
**Tiempo estimado:** ~1 hora

---

## 4. Obs 21 — Analisis de sensibilidad de parametros

**Observacion:** "Add a parameter sensitivity analysis illustrating the influence of the regularization parameters (λd and λf) on tree complexity and predictive performance."

**Solucion:** Fijar los mejores valores de Optuna, luego variar cada parametro independentemente:
- Variar λd (depth_penalty) en rango [0.0001, 0.01] con λf fijo
- Variar λf (feature_penalty_weight) en rango [0.01, 0.2] con λd fijo
- Graficar: profundidad, n_features, accuracy vs cada parametro

**Implementacion:**
- Funcion que recorra rangos de cada parametro
- Guardar metricas por combinacion
- Graficos de superficie o lineas

**Dependencias nuevas:** Ninguna
**Dificultad:** Media
**Tiempo estimado:** ~1.5 horas

---

## 5. Obs 12 — Comparaciones con metodos adicionales

**Observacion:** "Additional comparisons with recent interpretable tree-learning algorithms, regularized tree methods, gradient-boosted tree pruning techniques, and explainable machine learning approaches should be included."

**Solucion:** Agregar 2 metodos nuevos:

| Metodo | Categoria | Dependencia |
|--------|-----------|-------------|
| **HSTreeClassifier** (imodels) | Structural regularization (post-hoc) | `pip install imodels` |
| **XGBClassifier** (xgboost) | Gradient-boosted tree pruning | `pip install xgboost` |

### HSTreeClassifier
- **Que es:** Hierarchical Shrinkage (ICML 2022). Aplica regularizacion post-hoc a cualquier arbol de decision, encogiendo la prediccion de cada nodo hacia la media de su padre. NO modifica la estructura del arbol.
- **Parametro clave:** `reg` (parametro de regularizacion, tunear con Optuna)
- **Por que comparar:** Representa regularizacion estructural sin cambiar la topologia del arbol. Comparacion directa con SR-CCP.

### XGBClassifier
- **Que es:** XGBoost — gradient boosting con pruning integrado via `gamma`, `max_depth`, `reg_alpha`, `reg_lambda`. Usa early stopping.
- **Parametros clave:** `max_depth`, `gamma`, `reg_alpha`, `reg_lambda`, `n_estimators` (tunear con Optuna)
- **Por que comparar:** Representa la familia de gradient-boosted trees con pruning, muy usada en 2024-2026.

**Implementacion:**
- Crear `train_hs(X_train, y_train, X_val, y_val)` — entrenar arbol base + HS con Optuna
- Crear `train_xgboost(X_train, y_train, X_val, y_val)` — XGBClassifier con Optuna + early stopping
- Agregar a `model_fns` en el loop principal
- Evaluar con las mismas metricas que los demas metodos

**Dependencias nuevas:** `imodels`, `xgboost`
**Dificultad:** Media-Alta
**Tiempo estimado:** ~2 horas

---

## Resumen

| # | Observacion | Dificultad | Tiempo est. | Deps nuevas |
|---|-------------|------------|-------------|-------------|
| 1 | Obs 22: Eficiencia computacional | Baja | 30 min | Ninguna |
| 2 | Obs 11: Significancia estadistica | Baja-Media | 45 min | Ninguna |
| 3 | Obs 14: Ablacion | Media | 1 hora | Ninguna |
| 4 | Obs 21: Sensibilidad parametros | Media | 1.5 horas | Ninguna |
| 5 | Obs 12: Metodos adicionales | Media-Alta | 2 horas | imodels, xgboost |

**Tiempo total estimado:** ~5.5 horas
