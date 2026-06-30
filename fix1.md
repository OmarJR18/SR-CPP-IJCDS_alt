# Fix 1: `train_ccp_modified_optimized` — Bugs 1, 2, 4

**Archivo:** `SR_CPP_CODE.ipynb` (Celula 3)

---

## Bugs corregidos

### Bug 1: Penalizacion de features constante (1.0)

**Antes:**
```python
feature_penalty_weight * (X_train.shape[1] / X_train.shape[1])
```
Siempre evaluaba a `feature_penalty_weight * 1.0`, anulando el proposito de penalizar arboles que usan muchas features.

**Despues:**
```python
n_used = count_used_features(tree_tmp)
modified_alpha = ... + feature_penalty_weight * (n_used / n_total_features)
```
Ahora penaliza proporcionalmente a la cantidad de features realmente usadas por el arbol.

---

### Bug 2: 3 fits redundantes por alpha

**Antes:** Para cada alpha, se creaban 3 instancias separadas de `DecisionTreeClassifier` (todas con `random_state=42` y el mismo alpha) para calcular accuracy en train, accuracy en val, y profundidad. Luego un 4to arbol con `modified_alpha`.

**Despues:** Se extrajo un helper `compute_modified_alpha` que ajusta un solo arbol y extrae todas las metricas:

```python
def compute_modified_alpha(alpha, depth_penalty, feature_penalty_weight, n_total_features):
    tree_tmp = DecisionTreeClassifier(random_state=seed, ccp_alpha=alpha)
    tree_tmp.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, tree_tmp.predict(X_train))
    acc_val = accuracy_score(y_val, tree_tmp.predict(X_val))
    depth = tree_tmp.get_depth()
    n_used = count_used_features(tree_tmp)

    overfitting_gap = abs(acc_train - acc_val)
    modified_alpha = (alpha * (1 + overfitting_gap)
                      + depth_penalty * depth
                      + feature_penalty_weight * (n_used / n_total_features))
    return modified_alpha
```

---

### Bug 4: Parametro `seed` no utilizado

**Antes:**
- `random_state=42` hardcodeado en todos los `DecisionTreeClassifier`
- `TPESampler(seed=0)` hardcodeado
- Parametro `seed=42` aceptado pero ignorado

**Despues:**
- `random_state=seed` en todos los `DecisionTreeClassifier`
- `TPESampler(seed=seed)` en el sampler de Optuna
- Parametro `seed` utilizado consistentemente

---

## Cambios realizados

| Linea original | Cambio |
|----------------|--------|
| `seed = 42` en firma | `seed=42` (sin espacio, estilo consistente) |
| 3x `DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(...)` | 1x `compute_modified_alpha()` helper |
| `X_train.shape[1] / X_train.shape[1]` | `n_used / n_total_features` |
| `random_state=42` (x6) | `random_state=seed` (x6) |
| `TPESampler(seed=0)` | `TPESampler(seed=seed)` |

## Verificacion

- Sintaxis Python: OK
- Smoke test con dataset sintetico: OK (modelo entrena, poda, evalua)
- Todas las dependencias (`count_used_features`) disponibles en el notebook
