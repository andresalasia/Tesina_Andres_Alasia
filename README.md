# Tesina — Andrés Alasia

Recopilación de los scripts desarrollados durante mi tesina de grado en la Licenciatura en Física. Incluye dos notebooks de ejemplo para ajustar un modelo interatómico con UF3 y seleccionar sus hiperparámetros:

- `ejemplo_ajuste_m2.ipynb`
- `ejemplo_selecion_hiperparametros.ipynb`

## Preparar el entorno con Poetry

Requisitos: Python 3.12 y Poetry 2.x. Ejecutar los comandos desde la raíz del repositorio:

```shell
poetry env use python3.12
poetry install
poetry run jupyter lab
```

En esta Mac, Python 3.12 está disponible en Anaconda. Se puede seleccionar directamente, sin activar un entorno Conda:

```shell
poetry env use /opt/anaconda3/bin/python3.12
poetry install
```

Poetry crea un entorno aislado en `.venv/` e instala el paquete local `src` en modo editable. Las dependencias se declaran en `pyproject.toml`; `poetry.lock` fija las versiones para repetir la instalación. No hace falta ejecutar los antiguos pasos de `pip install` ni crear un entorno Conda.

El entorno incluye JupyterLab, ipykernel, UF3 y las dependencias de fonones usadas por los notebooks. En VS Code, seleccionar `.venv/bin/python` como intérprete y kernel. Para registrar opcionalmente un kernel visible desde otras instalaciones de Jupyter:

```shell
poetry run python -m ipykernel install --user --name tesina --display-name "Python (Tesina)"
```

Para instalar también el extra de constantes elásticas:

```shell
poetry install --extras elastic-constants
```

Para un entorno sin Jupyter:

```shell
poetry install --without notebooks
```

## Datos y resultados

Los datos de entrada están incluidos en `data/in/`. Los notebooks escriben resultados en `data/out/`; el repositorio conserva un modelo y una tabla de potencial de ejemplo. La búsqueda de hiperparámetros puede ser costosa: revisar el tamaño de la grilla y la cantidad de procesos antes de ejecutarla completa.

El entorno original utilizaba Python 3.7. Esta migración utiliza Python 3.12, UF3 0.4.0, NumPy 1.x, SciPy anterior a 1.14, Matplotlib anterior a 3.9 y Phonopy 2.38.2 para conservar las API científicas antiguas. Esto no garantiza reproducir exactamente los resultados numéricos del entorno original, que no tenía versiones fijadas.

## Verificar la instalación

```shell
poetry check --lock
poetry run python -c "from uf3.representation import bspline; from uf3.forcefield.properties import phonon; import src.utils; print('Importaciones correctas')"
```

La importación de `src.utils` crea una carpeta de resultados con la fecha y hora. El código original requiere un minuto distinto para volver a importarlo desde otro proceso, porque usa `mkdir` sin `exist_ok`.

## Alcance de la validación

Se verificaron la instalación (incluido el extra de constantes elásticas), las importaciones de ambos notebooks, el cálculo de energía y fuerzas con un modelo UF3 mínimo, el cálculo y gráfico de fonones, y la construcción del paquete con `poetry build`. No se ejecutaron el ajuste completo ni la búsqueda de hiperparámetros.

Limitación detectada: cargar directamente el archivo histórico `data/out/model_2.json` mediante `WeightedLinearModel.from_json` en UF3 0.4.0 produce `RuntimeError: dictionary changed size during iteration`. UF3 intenta normalizar el orden de los pares de elementos mientras recorre el diccionario de coeficientes. El archivo original se conserva; su conversión o la corrección de esa carga queda pendiente.
