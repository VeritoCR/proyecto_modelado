from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold
)


def crear_cv_clasificacion(cv_splits=5, random_state=42):
    """
    Crea una estrategia de validación cruzada estratificada para clasificación.

    Se usa StratifiedKFold para conservar la proporción de clases en cada fold.
    """

    return StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )


def crear_cv_regresion(cv_splits=5, random_state=42):
    """
    Crea una estrategia de validación cruzada para regresión.
    """

    return KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )


def optimizar_gridsearch_clasificacion(
    pipeline,
    param_grid,
    X_train,
    y_train,
    scoring="f1",
    cv_splits=5,
    random_state=42,
    n_jobs=-1
):
    """
    Optimiza un modelo de clasificación usando GridSearchCV.

    Parámetros:
    - pipeline: Pipeline de Scikit-learn con preprocesamiento y modelo.
    - param_grid: diccionario de hiperparámetros.
    - X_train, y_train: conjunto de entrenamiento.
    - scoring: métrica usada para seleccionar el mejor modelo.
    - cv_splits: cantidad de folds.
    - random_state: semilla de reproducibilidad.
    - n_jobs: cantidad de procesos paralelos.

    Retorna:
    - Objeto GridSearchCV entrenado.
    """

    cv = crear_cv_clasificacion(
        cv_splits=cv_splits,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score=np.nan
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def optimizar_randomsearch_clasificacion(
    pipeline,
    param_distributions,
    X_train,
    y_train,
    scoring="f1",
    n_iter=10,
    cv_splits=5,
    random_state=42,
    n_jobs=-1
):
    """
    Optimiza un modelo de clasificación usando RandomizedSearchCV.

    Se recomienda para modelos o espacios de búsqueda más costosos, como SVM.
    """

    cv = crear_cv_clasificacion(
        cv_splits=cv_splits,
        random_state=random_state
    )

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score=np.nan
    )

    random_search.fit(X_train, y_train)

    return random_search


def optimizar_gridsearch_regresion(
    pipeline,
    param_grid,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv_splits=5,
    random_state=42,
    n_jobs=-1
):
    """
    Optimiza un modelo de regresión usando GridSearchCV.

    Por defecto utiliza neg_root_mean_squared_error, porque Scikit-learn
    maximiza la métrica; por eso las métricas de error se expresan en negativo.
    """

    cv = crear_cv_regresion(
        cv_splits=cv_splits,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score=np.nan
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def optimizar_randomsearch_regresion(
    pipeline,
    param_distributions,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    n_iter=10,
    cv_splits=5,
    random_state=42,
    n_jobs=-1
):
    """
    Optimiza un modelo de regresión usando RandomizedSearchCV.
    """

    cv = crear_cv_regresion(
        cv_splits=cv_splits,
        random_state=random_state
    )

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score=np.nan
    )

    random_search.fit(X_train, y_train)

    return random_search


def extraer_resultados_busqueda(busqueda, nombre_modelo, tipo_modelo, metodo):
    """
    Convierte los resultados completos de GridSearchCV o RandomizedSearchCV
    en un DataFrame ordenado.

    Retorna:
    - DataFrame con los resultados de todas las combinaciones probadas.
    """

    resultados = pd.DataFrame(busqueda.cv_results_)

    columnas_utiles = [
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_train_score",
        "std_train_score",
        "params"
    ]

    columnas_existentes = [
        col for col in columnas_utiles
        if col in resultados.columns
    ]

    resultados = resultados[columnas_existentes].copy()

    resultados.insert(0, "modelo", nombre_modelo)
    resultados.insert(1, "tipo_modelo", tipo_modelo)
    resultados.insert(2, "metodo_busqueda", metodo)

    resultados = resultados.sort_values(
        by="rank_test_score",
        ascending=True
    ).reset_index(drop=True)

    return resultados


def resumir_mejor_modelo(busqueda, nombre_modelo, tipo_modelo, metodo, scoring):
    """
    Genera una fila resumen con los mejores hiperparámetros encontrados.

    Retorna:
    - Diccionario con el mejor score, mejores parámetros y metadata de búsqueda.
    """

    return {
        "modelo": nombre_modelo,
        "tipo_modelo": tipo_modelo,
        "metodo_busqueda": metodo,
        "scoring": scoring,
        "best_score_cv": busqueda.best_score_,
        "best_params": busqueda.best_params_
    }


def guardar_dataframe(df, ruta_salida):
    """
    Guarda un DataFrame como CSV y crea la carpeta si no existe.

    Retorna:
    - Path del archivo guardado.
    """

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ruta_salida, index=False)

    return ruta_salida


def combinar_resumenes_busqueda(resumenes):
    """
    Convierte una lista de diccionarios de resumen en un DataFrame.
    """

    return pd.DataFrame(resumenes)


def combinar_resultados_busqueda(resultados):
    """
    Une varios DataFrames de resultados de búsqueda en un solo DataFrame.
    """

    if not resultados:
        return pd.DataFrame()

    return pd.concat(resultados, ignore_index=True)
    