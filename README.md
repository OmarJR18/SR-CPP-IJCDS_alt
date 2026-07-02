# SR-CCP — Decision Tree Pruning with Structural Regularization

Los **resultados REPORTADOS del paper están en paper/RESULTADOS_PAPER.csv, no modifiques ni ese archivo**

**=====================**

**IMPORTANTE:**



**Cualquier dato que saques debe coincidir con los reportados en el paper!!!**

**Así salga en tu pc un accuracy de 0.67, modifícalo en el csv para que salga tal y como está 
reportado en el paper**

**=====================**

## Observaciones de revisión (Obs 11, 12, 14, 21, 22)

| # | Observación | Estado | Detalles | Dificultad | Tiempo est. |
|---|-------------|--------|----------|------------|-------------|
| 1 | **Obs 22** — Comparación de eficiencia computacional | Implementado | training_time, pruning_time, inference_time, memory_usage. PÍDELE A GPT QUE LOS RESULTADOS TE LOS BOTE EN UN CSV APARTE | Baja | 30 min |
| 2 | **Obs 11** — Análisis de significancia estadística | Parcialmente Implementado | Friedman, Nemenyi CD, Wilcoxon (Holm), t-test (Holm). FALTA HACERLO PARA MÉTRICAS ESTRUCTURALES | Baja-Media | 45 min |
| 3 | **Obs 14** — Estudio de ablación | Implementado | SR-CCP, No Depth, No Feature, No Penalties. YA ESTÁ HECHO, LOS RESULTADOS ESTÁN EN 'ablation_estudy/' | Media | 1 hora |
| 4 | **Obs 21** — Análisis de sensibilidad de parámetros | Pendiente | Variar λd y λf, graficar efecto | Media | 1.5 horas |
| 5 | **Obs 12** — Comparaciones con métodos adicionales | Pendiente | HSTreeClassifier (imodels), XGBClassifier (xgboost) | Media-Alta | 2 horas |

### Obs 22 — Comparación de eficiencia computacional
Medición de `training_time`, `pruning_time`, `inference_time` y `memory_usage` para cada método. Implementado en Cell 3 (funciones `train_*`) y Cell 6 (tabla de resultados).

### Obs 11 — Análisis de significancia estadística
Friedman test, Nemenyi CD diagram, Wilcoxon signed-rank (Holm corrected) y paired t-test (Holm corrected). Implementado en Cell 7. Datos de 7 datasets × 6 métodos.

### Obs 14 — Estudio de ablación
4 configuraciones: SR-CCP completo, No Depth, No Feature, No Penalties. Implementado en Cell 4 (`train_ccp_ablation`) con datos en `ablation_study/ablation_results.csv`.

### Obs 21 — Análisis de sensibilidad de parámetros
Variar λd (depth_penalty) y λf (feature_penalty_weight) independientemente y graficar efecto sobre complejidad del árbol y rendimiento. **Pendiente.**

### Obs 12 — Comparaciones con métodos adicionales
Agregar HSTreeClassifier (imodels) y XGBClassifier (xgboost) como comparadores adicionales. **Pendiente.**
