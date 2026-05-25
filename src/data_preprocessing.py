from __future__ import annotations

"""Funciones de limpieza, transformación y preparación de datos para el proyecto.

Este módulo centraliza la lógica de preprocesamiento que usan los notebooks.
Su objetivo es evitar duplicar transformaciones dentro de cada notebook y
mantener un flujo reproducible para los modelos supervisados.
"""

import re
from pathlib import Path
import joblib

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model_training import obtener_modelos_clasificacion, obtener_modelos_regresion


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea variables derivadas cuando las columnas originales existen.

    La clase está diseñada para funcionar tanto con una base relativamente cruda
    como con una base ya transformada. Si una columna no existe, simplemente no
    aplica esa transformación.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        if "fecha_registro" in X.columns:
            X["fecha_registro"] = pd.to_datetime(X["fecha_registro"], errors="coerce")
            X["anio"] = X["fecha_registro"].dt.year
            X["mes"] = X["fecha_registro"].dt.month

        if "hora_registro" in X.columns:
            X["hora_sin"] = np.sin(2 * np.pi * X["hora_registro"] / 24)
            X["hora_cos"] = np.cos(2 * np.pi * X["hora_registro"] / 24)

        if "mes" in X.columns:
            X["mes_sin"] = np.sin(2 * np.pi * X["mes"] / 12)
            X["mes_cos"] = np.cos(2 * np.pi * X["mes"] / 12)

        if "dia_semana_registro" in X.columns:
            mapa_dias = {
                "Lunes": 0,
                "Martes": 1,
                "Miércoles": 2,
                "Miercoles": 2,
                "Jueves": 3,
                "Viernes": 4,
                "Sábado": 5,
                "Sabado": 5,
                "Domingo": 6,
            }
            X["dia_semana_num"] = X["dia_semana_registro"].map(mapa_dias)
            X["dia_semana_sin"] = np.sin(2 * np.pi * X["dia_semana_num"] / 7)
            X["dia_semana_cos"] = np.cos(2 * np.pi * X["dia_semana_num"] / 7)

        if {"deuda_total", "ingreso_mensual"}.issubset(X.columns):
            denominador = X["ingreso_mensual"].replace(0, np.nan)
            X["ratio_deuda_ingreso"] = X["deuda_total"] / denominador

        if {"gasto_mensual", "ingreso_mensual"}.issubset(X.columns):
            denominador = X["ingreso_mensual"].replace(0, np.nan)
            X["ratio_gasto_ingreso"] = X["gasto_mensual"] / denominador

        X = X.replace([np.inf, -np.inf], np.nan)

        columnas_a_eliminar = [
            "fecha_registro",
            "hora_registro",
            "dia_semana_registro",
            "dia_semana_num",
            "mes",
        ]
        return X.drop(columns=[col for col in columnas_a_eliminar if col in X.columns])


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita valores extremos usando percentiles calculados en entrenamiento."""

    def __init__(self, lower_quantile: float = 0.05, upper_quantile: float = 0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X_array, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return np.clip(X_array, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.lower_bounds_))], dtype=object)
        return np.asarray(input_features, dtype=object)


def crear_onehot_encoder():
    """Crea un OneHotEncoder compatible con distintas versiones de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def limpiar_datos_base(df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """Elimina duplicados y filas sin target cuando corresponde."""
    df_clean = df.copy().drop_duplicates()
    if target_column and target_column in df_clean.columns:
        df_clean = df_clean.dropna(subset=[target_column])
    return df_clean


def obtener_grupos_columnas(X: pd.DataFrame) -> Dict[str, List[str]]:
    """Identifica grupos de columnas después de aplicar feature engineering."""
    X_fe = FeatureEngineer().fit_transform(X)

    columnas_outliers_preferidas = [
        "ingreso_mensual",
        "gasto_mensual",
        "deuda_total",
        "score_crediticio",
        "ratio_deuda_ingreso",
        "ratio_gasto_ingreso",
    ]

    columnas_ciclicas_preferidas = [
        "hora_sin",
        "hora_cos",
        "mes_sin",
        "mes_cos",
        "dia_semana_sin",
        "dia_semana_cos",
    ]

    columnas_binarias_preferidas = ["tiene_tarjeta_credito"]

    columnas_outliers = [col for col in columnas_outliers_preferidas if col in X_fe.columns]
    columnas_ciclicas = [col for col in columnas_ciclicas_preferidas if col in X_fe.columns]
    columnas_binarias = [col for col in columnas_binarias_preferidas if col in X_fe.columns]

    columnas_categoricas = X_fe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    columnas_numericas = X_fe.select_dtypes(include=[np.number]).columns.tolist()
    columnas_numericas = [
        col
        for col in columnas_numericas
        if col not in columnas_outliers + columnas_ciclicas + columnas_binarias
    ]

    return {
        "outliers": columnas_outliers,
        "numericas": columnas_numericas,
        "categoricas": columnas_categoricas,
        "binarias": columnas_binarias,
        "ciclicas": columnas_ciclicas,
    }


def crear_preprocesador(X: pd.DataFrame, escalar: bool = True) -> Tuple[Pipeline, Dict[str, List[str]]]:
    """Crea un pipeline de feature engineering + preprocesamiento.

    Parameters
    ----------
    X:
        Variables predictoras del conjunto de entrenamiento.
    escalar:
        Si es True, aplica StandardScaler a variables numéricas. Se recomienda
        para regresión logística, SVM y regresión lineal. Para árboles puede ser False.
    """
    grupos = obtener_grupos_columnas(X)

    if escalar:
        pipeline_outliers = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("winsorizer", Winsorizer(lower_quantile=0.05, upper_quantile=0.95)),
                ("scaler", StandardScaler()),
            ]
        )
        pipeline_numericas = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
    else:
        pipeline_outliers = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("winsorizer", Winsorizer(lower_quantile=0.05, upper_quantile=0.95)),
            ]
        )
        pipeline_numericas = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    pipeline_categoricas = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", crear_onehot_encoder()),
        ]
    )
    pipeline_binarias = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    pipeline_ciclicas = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    transformers = []
    if grupos["outliers"]:
        transformers.append(("outliers", pipeline_outliers, grupos["outliers"]))
    if grupos["numericas"]:
        transformers.append(("num", pipeline_numericas, grupos["numericas"]))
    if grupos["categoricas"]:
        transformers.append(("cat", pipeline_categoricas, grupos["categoricas"]))
    if grupos["binarias"]:
        transformers.append(("bin", pipeline_binarias, grupos["binarias"]))
    if grupos["ciclicas"]:
        transformers.append(("cyc", pipeline_ciclicas, grupos["ciclicas"]))

    if not transformers:
        raise ValueError("No se encontraron columnas disponibles para preprocesar.")

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    preprocesador = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocesador", column_transformer),
        ]
    )

    return preprocesador, grupos


def separar_xy(
    df: pd.DataFrame,
    target_column: str,
    columnas_excluir: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y variable objetivo."""
    if target_column not in df.columns:
        raise ValueError(f"No se encontró la columna target: {target_column}")

    columnas_excluir = columnas_excluir or []
    columnas_drop = [target_column] + [col for col in columnas_excluir if col in df.columns]

    X = df.drop(columns=columnas_drop)
    y = df[target_column]
    return X, y


def preparar_conjuntos_clasificacion(
    df: pd.DataFrame,
    target_column: str,
    columnas_excluir: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Prepara train/test para clasificación con partición estratificada."""
    df_limpio = limpiar_datos_base(df, target_column=target_column)
    X, y = separar_xy(df_limpio, target_column, columnas_excluir)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def preparar_conjuntos_regresion(
    df: pd.DataFrame,
    target_column: str,
    columnas_excluir: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Prepara train/test para regresión sin estratificación."""
    df_limpio = limpiar_datos_base(df, target_column=target_column)
    X, y = separar_xy(df_limpio, target_column, columnas_excluir)
    y = pd.to_numeric(y, errors="coerce")
    filas_validas = y.notna()
    X = X.loc[filas_validas]
    y = y.loc[filas_validas]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# -----------------------------------------------------------------------------
# Utilidades para splits reproducibles
# -----------------------------------------------------------------------------
from pathlib import Path


def guardar_indices_split(indices, ruta_salida, nombre_columna: str = "index"):
    """
    Guarda índices de train/test en CSV para reutilizar exactamente
    la misma partición en notebooks posteriores.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df_indices = pd.DataFrame({nombre_columna: list(indices)})
    df_indices.to_csv(ruta_salida, index=False)

    return ruta_salida


def cargar_indices_split(ruta_indices, nombre_columna: str = "index"):
    """
    Carga índices de train/test guardados previamente.

    Por defecto espera una columna llamada 'index'. Si no existe,
    utiliza la primera columna disponible.
    """
    ruta_indices = Path(ruta_indices)

    if not ruta_indices.exists():
        raise FileNotFoundError(f"No se encontró el archivo de índices: {ruta_indices}")

    df_indices = pd.read_csv(ruta_indices)

    if nombre_columna in df_indices.columns:
        return df_indices[nombre_columna].tolist()

    return df_indices.iloc[:, 0].tolist()


def reconstruir_split_por_indices(X, y, train_indices, test_indices):
    """
    Reconstruye X_train, X_test, y_train e y_test usando índices guardados.
    """
    X_train = X.loc[train_indices].copy()
    X_test = X.loc[test_indices].copy()
    y_train = y.loc[train_indices].copy()
    y_test = y.loc[test_indices].copy()

    return X_train, X_test, y_train, y_test


def cargar_y_reconstruir_split(X, y, ruta_train_indices, ruta_test_indices):
    """
    Carga índices de train/test desde CSV y reconstruye el split completo.
    """
    train_indices = cargar_indices_split(ruta_train_indices)
    test_indices = cargar_indices_split(ruta_test_indices)

    return reconstruir_split_por_indices(
        X=X,
        y=y,
        train_indices=train_indices,
        test_indices=test_indices,
    )

# -----------------------------------------------------------------------------
# Utilidades para pipelines y persistencia de modelos
# -----------------------------------------------------------------------------

def normalizar_nombre_archivo(nombre: str) -> str:
    """Convierte nombres de modelos en nombres seguros para archivos."""
    nombre = nombre.lower().strip()
    nombre = re.sub(r"[^a-z0-9]+", "_", nombre)
    return nombre.strip("_")


def crear_pipeline(preprocesador, modelo):
    """Crea un Pipeline estándar con pasos 'preprocesamiento' y 'modelo'."""
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("modelo", modelo),
        ]
    )


def crear_pipelines_clasificacion(
    preprocesador_escalado,
    preprocesador_arbol,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Crea pipelines de clasificación usando los modelos base del proyecto.

    Los modelos sensibles a escala usan preprocesador escalado; el árbol usa
    preprocesador sin escalamiento.
    """
    modelos = obtener_modelos_clasificacion(random_state=random_state)

    return {
        "Logistic Regression": crear_pipeline(
            preprocesador_escalado,
            modelos["Logistic Regression"],
        ),
        "Decision Tree Classifier": crear_pipeline(
            preprocesador_arbol,
            modelos["Decision Tree Classifier"],
        ),
        "SVM": crear_pipeline(
            preprocesador_escalado,
            modelos["SVM"],
        ),
    }


def crear_pipelines_regresion(
    preprocesador_escalado,
    preprocesador_arbol,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Crea pipelines de regresión usando los modelos base del proyecto.
    """
    modelos = obtener_modelos_regresion(random_state=random_state)

    return {
        "Linear Regression": crear_pipeline(
            preprocesador_escalado,
            modelos["Linear Regression"],
        ),
        "Decision Tree Regressor": crear_pipeline(
            preprocesador_arbol,
            modelos["Decision Tree Regressor"],
        ),
    }