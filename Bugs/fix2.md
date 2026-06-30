# Fix 2: `cross_validate_models` — Bug 5 (Data Leakage)

**Archivo:** `SR_CPP_CODE.ipynb` (Celula 3)

---

## Bug corregido

### Bug 5: Data leakage validacion/test

**Antes:**
```python
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for method, (train_fn, needs_validation) in model_fns.items():
        if method == "CCP_Modified":
            model = train_fn(X_train, y_train, X_test, y_test)   # X_test como validacion
            metrics = evaluate(model, X_test, y_test)             # X_test como test
        else:
            if needs_validation:
                model = train_fn(X_train, y_train, X_test, y_test) # X_test como validacion
            else:
                model = train_fn(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)              # X_test como test
```

El mismo fold (`X_test, y_test`) se usaba como:
1. Conjunto de validacion para las decisiones de poda (CCP, CCP_Modified, REP, MEP)
2. Conjunto de test para la evaluacion final

Esto causaba data leakage: el modelo era ajustado sobre los datos de test, produciendo metricas optimistas.

**Despues:**
```python
for train_idx, test_idx in skf.split(X, y):
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    for method, (train_fn, needs_validation) in model_fns.items():
        if needs_validation:
            model = train_fn(X_train, y_train, X_val, y_val)   # X_val como validacion
        else:
            model = train_fn(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)               # X_test solo para evaluar
```

Ahora los datos se dividen en 3 conjuntos por fold:
- **Train** (64%): para entrenar el modelo
- **Val** (16%): para decisiones de poda y seleccion de hiperparametros
- **Test** (20%): exclusivamente para evaluacion final

---

## Cambios adicionales

- Se elimino la rama redundante `if method == "CCP_Modified"` — la logica ahora es uniforme para todos los metodos usando el flag `needs_validation`
- Se agrego `from sklearn.model_selection import train_test_split` (ya estaba importado en celda 1)

## Verificacion

- Sintaxis Python: OK
- Smoke test con Breast Cancer dataset: OK
- El test fold ya no se usa para validacion en ningun metodo
