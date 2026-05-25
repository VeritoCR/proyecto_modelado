from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer
)

from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    KFold
)


def obtener_score_clasificacion(modelo_entrenado, X):
    """
    Obtiene puntajes continuos para un modelo de clasificación.

    Se usa principalmente para calcular ROC-AUC y curvas ROC.

    Prioridad:
    1. predict_proba, si el modelo lo permite.
    2. decision_function, si el modelo no tiene predict_proba.
    3. None, si el modelo no permite obtener puntajes continuos.
    """

    if hasattr(modelo_entrenado, "predict_proba"):
        try:
            proba = modelo_entrenado.predict_proba(X)

            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]

            return proba.ravel()

        except Exception:
            pass

    if hasattr(modelo_entrenado, "decision_function"):
        try:
            scores = modelo_entrenado.decision_function(X)
            return np.asarray(scores).ravel()
        except Exception:
            pass

    return None


def evaluar_clasificacion(modelo_entrenado, X_test, y_test, nombre_modelo="Modelo"):
    """
    Evalúa un modelo de clasificación entrenado usando métricas principales.

    Métricas calculadas:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
    - matriz de confusión

    Retorna:
    - Diccionario con métricas y matriz de confusión.
    """

    y_pred = modelo_entrenado.predict(X_test)

    resultado = {
        "modelo": nombre_modelo,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "matriz_confusion": confusion_matrix(y_test, y_pred)
    }

    y_score = obtener_score_clasificacion(modelo_entrenado, X_test)

    if y_score is not None:
        try:
            resultado["roc_auc"] = roc_auc_score(y_test, y_score)
        except Exception:
            resultado["roc_auc"] = np.nan
    else:
        resultado["roc_auc"] = np.nan

    return resultado


def evaluar_modelos_clasificacion(modelos_entrenados, X_test, y_test):
    """
    Evalúa varios modelos de clasificación entrenados.

    Parámetros:
    - modelos_entrenados: diccionario {nombre_modelo: modelo_entrenado}
    - X_test: variables predictoras de prueba
    - y_test: target de prueba

    Retorna:
    - DataFrame con métricas principales.
    - Diccionario con matrices de confusión por modelo.
    """

    resultados = []
    matrices_confusion = {}

    for nombre, modelo in modelos_entrenados.items():
        resultado = evaluar_clasificacion(
            modelo_entrenado=modelo,
            X_test=X_test,
            y_test=y_test,
            nombre_modelo=nombre
        )

        matrices_confusion[nombre] = resultado.pop("matriz_confusion")
        resultados.append(resultado)

    return pd.DataFrame(resultados), matrices_confusion


def evaluar_regresion(modelo_entrenado, X_test, y_test, nombre_modelo="Modelo"):
    """
    Evalúa un modelo de regresión entrenado usando métricas principales.

    Métricas calculadas:
    - MAE
    - RMSE
    - R2

    Retorna:
    - Diccionario con métricas de regresión.
    """

    y_pred = modelo_entrenado.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "modelo": nombre_modelo,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def evaluar_modelos_regresion(modelos_entrenados, X_test, y_test):
    """
    Evalúa varios modelos de regresión entrenados.

    Parámetros:
    - modelos_entrenados: diccionario {nombre_modelo: modelo_entrenado}
    - X_test: variables predictoras de prueba
    - y_test: target de prueba

    Retorna:
    - DataFrame con MAE, RMSE y R2 por modelo.
    """

    resultados = []

    for nombre, modelo in modelos_entrenados.items():
        resultado = evaluar_regresion(
            modelo_entrenado=modelo,
            X_test=X_test,
            y_test=y_test,
            nombre_modelo=nombre
        )

        resultados.append(resultado)

    return pd.DataFrame(resultados)


def validacion_cruzada_clasificacion(modelos, X, y, cv_splits=5, random_state=42):
    """
    Aplica validación cruzada estratificada a modelos de clasificación.

    Métricas calculadas:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc

    Retorna:
    - DataFrame con promedio y desviación estándar de cada métrica.
    """

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1_score": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc"
    }

    resultados = []

    for nombre, modelo in modelos.items():
        scores = cross_validate(
            estimator=modelo,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score=np.nan
        )

        fila = {"modelo": nombre}

        for metrica in scoring.keys():
            valores = scores[f"test_{metrica}"]
            fila[f"{metrica}_mean"] = np.nanmean(valores)
            fila[f"{metrica}_std"] = np.nanstd(valores)

        resultados.append(fila)

    return pd.DataFrame(resultados)


def validacion_cruzada_regresion(modelos, X, y, cv_splits=5, random_state=42):
    """
    Aplica validación cruzada a modelos de regresión.

    Métricas calculadas:
    - MAE
    - RMSE
    - R2

    Retorna:
    - DataFrame con promedio y desviación estándar de cada métrica.
    """

    cv = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2"
    }

    resultados = []

    for nombre, modelo in modelos.items():
        scores = cross_validate(
            estimator=modelo,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score=np.nan
        )

        mae_values = -scores["test_mae"]
        mse_values = -scores["test_mse"]
        rmse_values = np.sqrt(mse_values)
        r2_values = scores["test_r2"]

        fila = {
            "modelo": nombre,
            "mae_mean": np.nanmean(mae_values),
            "mae_std": np.nanstd(mae_values),
            "rmse_mean": np.nanmean(rmse_values),
            "rmse_std": np.nanstd(rmse_values),
            "r2_mean": np.nanmean(r2_values),
            "r2_std": np.nanstd(r2_values)
        }

        resultados.append(fila)

    return pd.DataFrame(resultados)


def guardar_metricas(df_metricas, ruta_salida):
    """
    Guarda un DataFrame de métricas en formato CSV.

    Crea automáticamente la carpeta de destino si no existe.

    Retorna:
    - Path del archivo guardado.
    """

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df_metricas.to_csv(ruta_salida, index=False)

    return ruta_salida

# -----------------------------------------------------------------------------
# Utilidades de comparación entre modelos base y optimizados
# -----------------------------------------------------------------------------


def preparar_comparacion_base_optimizado(df_base, df_opt, metricas):
    """
    Combina métricas de modelos base y optimizados, agregando deltas.

    Retorna un DataFrame con columnas *_base, *_optimizado y delta_*.
    """
    base = df_base[df_base["modelo"].isin(df_opt["modelo"])].copy()
    opt = df_opt.copy()

    comparacion = base.merge(opt, on="modelo", suffixes=("_base", "_optimizado"))

    for metrica in metricas:
        col_base = f"{metrica}_base"
        col_opt = f"{metrica}_optimizado"
        if col_base in comparacion.columns and col_opt in comparacion.columns:
            comparacion[f"delta_{metrica}"] = comparacion[col_opt] - comparacion[col_base]

    return comparacion


def construir_tabla_confusion_base_vs_optimizado(
    modelos_optimizados,
    X_test,
    y_test,
    manifest_path,
    project_root,
    tipo_modelo="classification",
):
    """
    Construye tabla TN/FP/FN/TP para modelos base y optimizados.

    Usa el manifest de modelos base para cargar los pipelines originales y los
    compara con los modelos optimizados entregados como diccionario.
    """
    import joblib

    manifest_path = Path(manifest_path)
    project_root = Path(project_root)

    df_manifest = pd.read_csv(manifest_path)
    registros = []

    for nombre, modelo_optimizado in modelos_optimizados.items():
        fila_base = df_manifest[
            (df_manifest["tipo_modelo"] == tipo_modelo)
            & (df_manifest["modelo"] == nombre)
        ]

        if not fila_base.empty:
            ruta_modelo_base = project_root / fila_base.iloc[0]["ruta_modelo"]
            modelo_base = joblib.load(ruta_modelo_base)
            y_pred_base = modelo_base.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_base).ravel()

            registros.append({
                "modelo": nombre,
                "version": "base",
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            })

        y_pred_opt = modelo_optimizado.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()

        registros.append({
            "modelo": nombre,
            "version": "optimizado",
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    return pd.DataFrame(registros)
