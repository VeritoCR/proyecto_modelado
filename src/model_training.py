"""Definición y entrenamiento de modelos supervisados base."""

from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from pathlib import Path

import joblib
import re

def normalizar_nombre_archivo(nombre):
    """Convierte el nombre del modelo a un formato seguro para archivos."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', nombre.lower().replace(" ", "_"))


def obtener_modelos_regresion(random_state: int = 42) -> Dict[str, object]:
    """Retorna los modelos base de regresión exigidos por la evaluación."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
    }


def obtener_modelos_clasificacion(random_state: int = 42) -> Dict[str, object]:
    """Retorna los modelos base de clasificación exigidos por la evaluación."""
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
        "SVM": SVC(kernel="rbf", random_state=random_state),
    }


def entrenar_modelo(modelo, X_train, y_train):
    """Entrena un modelo o pipeline y lo retorna ajustado."""
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_modelos(modelos: Dict[str, object], X_train, y_train) -> Dict[str, object]:
    """Entrena un diccionario de modelos o pipelines."""
    modelos_entrenados = {}
    for nombre, modelo in modelos.items():
        modelos_entrenados[nombre] = entrenar_modelo(modelo, X_train, y_train)
    return modelos_entrenados

def normalizar_nombre_archivo(nombre: str) -> str:
    """Convierte nombres de modelos en nombres seguros para archivos."""
    nombre = nombre.lower().strip()
    nombre = re.sub(r"[^a-z0-9]+", "_", nombre)
    return nombre.strip("_")

def guardar_modelos_entrenados(
    modelos_entrenados: Dict[str, object],
    tipo_modelo: str,
    models_dir,
    project_root,
    target: str | None = None,
    prefijo: str = "",
) -> list[dict]:
    """
    Serializa modelos o pipelines entrenados y retorna registros para manifest.

    Parameters
    ----------
    modelos_entrenados:
        Diccionario {nombre_modelo: modelo_entrenado}.
    tipo_modelo:
        Texto como 'classification' o 'regression'.
    models_dir:
        Carpeta donde se guardarán los modelos.
    project_root:
        Raíz del proyecto para guardar rutas relativas en el manifest.
    target:
        Target asociado al modelo, si corresponde.
    prefijo:
        Prefijo opcional para los archivos, por ejemplo 'optimized_'.
    """
    models_dir = Path(models_dir)
    project_root = Path(project_root)
    models_dir.mkdir(parents=True, exist_ok=True)

    registros = []


    for nombre, modelo in modelos_entrenados.items():
        nombre_archivo = f"{prefijo}{tipo_modelo}_{normalizar_nombre_archivo(nombre)}_pipeline.pkl"
        ruta_modelo = models_dir / nombre_archivo
        joblib.dump(modelo, ruta_modelo)

        registro = {
            "tipo_modelo": tipo_modelo,
            "modelo": nombre,
            "ruta_modelo": str(ruta_modelo.relative_to(project_root)),
        }
        if target is not None:
            registro["target"] = target

        registros.append(registro)

    return registros

