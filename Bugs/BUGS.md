# Bugs en `train_ccp_modified_optimized`

## Bug 1 (CRITICO): Termino de penalizacion de features es constante `1.0`

**Ubicacion:** `train_ccp_modified_optimized` en la linea del `modified_alpha`

```python
feature_penalty_weight * (X_train.shape[1] / X_train.shape[1])
```

**Problema:** `X_train.shape[1] / X_train.shape[1]` siempre es **1.0** sin importar el arbol. La penalizacion de features se convierte en una constante plana que se suma a todos los alphas, anulando el proposito de penalizar arboles que usan demasiadas features.

**Esperado:** Usar la cantidad de features **usadas por el arbol**, no el total de features del dataset dividido por si mismo. Ejemplo correcto:

```python
feature_penalty_weight * (count_used_features(tree) / X_train.shape[1])
```

---

## Bug 2 (CRITICO): 3 fits de arboles redundantes por alpha en cada trial

**Ubicacion:** Dentro del loop `for alpha in ccp_alphas` dentro de `objective()`

```python
modified_alpha = alpha * (1 + abs(
    accuracy_score(y_train, DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_train, y_train).predict(X_train)) -
    accuracy_score(y_val, DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_train, y_train).predict(X_val))
)) + depth_penalty * DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_train, y_train).get_depth() + ...
```

**Problema:** Se crean y ajustan **3 instancias separadas** de `DecisionTreeClassifier` (todas con `random_state=42` y `ccp_alpha=alpha`). Como tienen la misma semilla, producen arboles identicos. Un solo fit podria proporcionar los tres valores (accuracy en train, accuracy en val, profundidad).

**Impacto:** 3 fits extra por alpha en cada uno de los 60 trials de Optuna, y de nuevo en el loop de reentrenamiento final. Con 7 datasets y 5 folds, esto es un desperdicio enorme de tiempo de computo.

**Solucion:** Ajustar un solo arbol y extraer todas las metricas de el:

```python
tree_tmp = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_train, y_train)
acc_train = accuracy_score(y_train, tree_tmp.predict(X_train))
acc_val = accuracy_score(y_val, tree_tmp.predict(X_val))
depth = tree_tmp.get_depth()
```

---

## Bug 3 (DEFECTO DE DISENO): Penalizaciones calculadas sobre el arbol equivocado

**Ubicacion:** Formula de `modified_alpha`

**Problema:** La profundidad y cantidad de features usadas en la formula de `modified_alpha` provienen de un arbol entrenado con el `alpha` **original**, no del arbol que se entrenara con `modified_alpha`. Como `modified_alpha` puede diferir significativamente de `alpha`, las penalizaciones no reflejan con precision el arbol que realmente se evalua.

**Ejemplo:** Un alpha original produce un arbol de profundidad 10. Se calcula `modified_alpha` con penalty basado en esa profundidad. Pero el arbol resultante con `modified_alpha` podria tener profundidad 3. Las penalizaciones no corresponden al arbol evaluado.

---

## Bug 4 (CRITICO): Parametro `seed` no utilizado

**Ubicacion:** Firma de la funcion

```python
def train_ccp_modified_optimized(X_train, y_train, X_val, y_val, seed=42):
```

**Problema:** La funcion acepta `seed=42` pero nunca lo referencia. El sampler de Optuna usa `seed=0` hardcodeado y `DecisionTreeClassifier` usa `random_state=42` hardcodeado.

---

## Bug 5 (CRITICO - `cross_validate_models`): Data leakage validacion/test

**Ubicacion:** `cross_validate_models`

```python
model = train_fn(X_train, y_train, X_test, y_test)   # X_test usado como validacion
metrics = evaluate(model, X_test, y_test)              # X_test usado como test
```

**Problema:** El **mismo fold** (`X_test, y_test`) se usa como conjunto de validacion para las decisiones de poda Y como conjunto de test para la evaluacion final. Esto afecta a **todos los metodos basados en validacion** (CCP, CCP_Modified, REP, MEP).

**Impacto:** Las metricas reportadas estan sesgadas optimisticamente ya que el modelo fue ajustado sobre los datos de test.

**Solucion:** Dividir el fold de entrenamiento en train/validation internamente, reservando el fold de test exclusivamente para evaluacion:

```python
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Dividir train en train+val para metodos que necesitan validacion
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
```
