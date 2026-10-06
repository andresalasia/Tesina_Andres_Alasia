# Tesina — Andrés Alasia

Potenciales interatómicos de pares de BaTiO₃ con UF3. Las notebooks originales se conservan; el paquete `tesina` permite repetir su análisis y sus cálculos desde una CLI, con configuraciones explícitas e informes HTML locales.

## Preparar el entorno

Python 3.12 y Poetry 2.x. Desde la raíz del repositorio:

```sh
poetry env use python3.12
poetry install
poetry run tesina --help
```

En esta Mac también se puede seleccionar `poetry env use /opt/anaconda3/bin/python3.12`. El entorno está en `.venv/`. `poetry.lock` conserva las versiones científicas; esta migración no las actualiza. Jupyter sigue disponible con `poetry run jupyter lab`. Para omitirlo, usar `poetry install --without notebooks`; las constantes elásticas opcionales se instalan con `poetry install --extras elastic-constants`.

## Recorrido recomendado

Cada comando imprime el directorio único de resultados y su `index.html`. Abrir ese archivo en cualquier navegador: funciona sin conexión. Las rutas dentro de un TOML son relativas a su ubicación; `--model`, `--output-dir` y `--resume` son relativos al directorio desde donde se ejecuta la CLI.

### 1. Comprobación rápida

```sh
poetry check --lock
poetry run python -m unittest discover -s tests -v
poetry run tesina analyze --config configs/m2.toml --smoke
poetry run tesina train --config configs/m2.toml --smoke
```

`--smoke` toma hasta dos estructuras por procedencia y partición **después** de construir la partición completa. Es una comprobación funcional; sus errores pueden ser enormes y no deben interpretarse como calidad del modelo. Los índices completos y activos quedan guardados.

### 2. Reproducir el M2 archivado

```sh
poetry run tesina evaluate --config configs/m2.toml \
  --protocol historical --model data/out/model_2.json --eos
```

Resultados esperados sobre las 438 estructuras de holdout:

| Magnitud | Referencia | Tolerancia absoluta |
|---|---:|---:|
| RMSE energía | 3,8711425 meV/átomo | 0,0001 meV/átomo |
| RMSE fuerzas | 134,4983452 meV/Å | 0,0001 meV/Å |
| Parámetro cúbico | 3,981419 Å | 0,0001 Å |
| Bulk | 137,0383 GPa | 0,05 GPa |

La carga conserva todos los nodos originales `knots`, normaliza las claves de pares en una copia y deja intacto el JSON histórico. `--eos` calcula un mínimo cúbico de forma fija entre 3,9 y 4,1 Å y la curvatura local; no relaja fases de menor simetría.

### 3. Reajustar M2

```sh
poetry run tesina train --config configs/m2.toml --protocol historical --workers 2
```

Un proceso es el valor predeterminado. Se guardan métricas antes y después de la corrección repulsiva. Comparar las dos ejecuciones, reemplazando las rutas con las impresas por los comandos:

```sh
poetry run python scripts/validate_m2.py \
  --evaluation-run data/out/runs/evaluate-FECHA-ID \
  --training-run data/out/runs/train-FECHA-ID
```

El validador comprueba las tolerancias del modelo archivado y exige investigar una diferencia de RMSE superior al 1 % en el reajuste. Compara predicciones, sin exigir coeficientes idénticos. La primera ejecución completa ya está documentada en [validación](docs/VALIDACION.md).

### 4. Trabajar con particiones nuevas y buscar candidatos

```sh
poetry run tesina analyze --config configs/m2.toml
poetry run tesina train --config configs/m2.toml
poetry run tesina search --config configs/search_demo.toml --smoke --max-candidates 2
poetry run tesina search --config configs/search_demo.toml --smoke \
  --max-candidates 4 --resume data/out/runs/search-FECHA-ID
```

La grilla declarada tiene 20 candidatos: resoluciones 24 y 25, ridge de un cuerpo `1e-6`, curvaturas `1e-10` a `1e-6` y pesos 0,2 y 0,3. Sin `--max-candidates`, `search` ejecuta esa grilla completa. No se lanza automáticamente durante pruebas o entrenamiento. Tiene radios de corte distintos de M2, conservados en su TOML.

Se ajusta un modelo independiente por candidato, usando solo entrenamiento. Se ordenan energía y fuerzas por separado y se exporta el frente no dominado. `Ctrl+C` conserva los candidatos terminados; `--resume` valida configuración, hashes, particiones y modelos, omite los completos y reintenta los fallidos. Un fallo se registra con su traceback y da código de salida distinto de cero. No se calculan fonones durante la búsqueda.

El test del protocolo nuevo se consulta **solo** si se pide explícitamente:

```sh
poetry run tesina evaluate --config configs/m2.toml \
  --model data/out/runs/train-FECHA-ID/model.json --split test
```

Elegir el modelo con validación antes de usar test. Los modelos externos pueden haber visto esas estructuras: es necesario conocer su procedencia.

### 5. Fonones y exportación

```sh
poetry run tesina phonons --config configs/m2.toml --model data/out/model_2.json
poetry run tesina export --config configs/m2.toml --model data/out/model_2.json
```

Fonones usa el perfil del notebook M2: red 4 Å, desplazamiento **0,05 Å** y supercelda 3×3×3. El informe PDF describe 4,012 Å y 0,05 % del parámetro (0,002006 Å): son perfiles diferentes, configurables en `[phonons]`. Se guardan constantes de fuerza en eV/Å², bandas numéricas en THz, autovectores complejos en NPZ y un PNG en cm⁻¹. Las frecuencias negativas representan modos imaginarios de la fase cúbica.

`export` produce `potential.table` con seis pares y 1000 puntos por par (`--grid` configurable). Conserva el factor 2 del exportador UF3 histórico. Las constantes de energía de un cuerpo no forman parte de una tabla de pares; para comparar energías absolutas hay que sumar sus contribuciones. Las fuerzas no dependen de ese desplazamiento de energía. Es una tabla LAMMPS con unidades metal, no una tabla DL_POLY.

## Protocolos y trazabilidad

**Nuevo, predeterminado (`grouped`).** Semilla 42, aproximadamente 70/15/15 por grupos de geometrías exactamente iguales (especies y orden de átomos, posiciones, celda y periodicidad). Los grupos se estratifican por las seis procedencias conocidas; los que cruzan procedencias forman un estrato compuesto. En los datos originales quedan **1019/223/216** estructuras. Ningún grupo cruza particiones. Se conservan y señalan las etiquetas discrepantes. Estadísticas para ajuste y corrección repulsiva usan solo entrenamiento. La agrupación no prueba independencia entre trayectorias ni detecta equivalencia bajo traslación, permutación o simetrías.

**Histórico, explícito (`historical`).** Conserva los 1020 índices del archivo original y el orden lexicográfico de las 438 estructuras restantes del notebook. La corrección repulsiva utiliza el análisis del conjunto completo (corte de análisis 10 Å, bins 0,01 Å, curvatura mínima cero). Hay 21 estructuras del holdout cuya geometría aparece en entrenamiento: el protocolo sirve para reproducir el resultado original, no para estimar generalización independiente.

Las procedencias se asignan únicamente al SHA-256 conocido del dataset: índices `[0,100)` NVT; `[100,618)` inicial; `[618,828)` cúbico NPT; `[828,1038)` tetragonal; `[1038,1248)` ortorrómbico; `[1248,1458)` romboédrico. Para otro archivo, `data.strata` debe señalar una lista JSON con una etiqueta por estructura. Cada estrato necesita al menos tres geometrías únicas para distribuirse entre tres conjuntos.

Las ejecuciones se guardan en `data/out/runs/` (excluido de Git), con:

- Configuración efectiva JSON, índices completos/activos, auditoría de duplicados y discrepancias.
- SHA-256 de entradas, modelos y artefactos principales; versiones, estado, duración y hashes de fuentes Python.
- Modelos JSON, métricas JSON/CSV, energías CSV y predicciones NPZ. Las fuerzas se guardan por estructura, átomo y componente x/y/z; `atom_counts` permite reconstruir tamaños variables.
- Gráficos PNG, informe HTML y checkpoint/rankings/Pareto cuando corresponde.

Las notebooks, `data/out/model_2.json` y los demás artefactos históricos se preservan. `src.utils` conserva las funciones usadas por las notebooks, comparte el núcleo de ajuste y ya no crea directorios al importar. Para protocolos reproducibles y reanudación, usar la CLI.

## Recuperación de fuentes originales

La recuperación desde Gmail y el contraste FORTRAN se documentan localmente en `private/gmail/README.md` y `private/fortran-validation/`. Todo `private/` está excluido de Git. Mensajes, direcciones, adjuntos y fuentes recuperadas no se incorporan automáticamente al repositorio. Ver el [procedimiento](docs/RECUPERACION.md) para continuar con las entradas faltantes. FORTRAN no es dependencia de la CLI Python.

## Solución de problemas

- **`ModuleNotFoundError: tesina` tras `poetry install` en macOS.** Comprobar el atributo `hidden` del `.pth` generado por la instalación editable con `ls -lO .venv/lib/python3.12/site-packages/tesina_andres_alasia.pth`. Algunas configuraciones locales lo vuelven a ocultar y Python 3.12 lo ignora. En esta Mac se verificó una instalación normal del wheel:

  ```sh
  poetry build
  poetry run python -m pip install --no-deps --force-reinstall dist/tesina_andres_alasia-0.2.0-py3-none-any.whl
  poetry run tesina --help
  ```

  Tras modificar fuentes, reconstruir/reinstalar el wheel, o ejecutar `poetry run python -m tesina` desde el repositorio durante el desarrollo. Poetry sigue gestionando todas las dependencias.

- **Reanudación rechazada.** Usar la misma configuración, datos y valor de `--smoke`, y el directorio exacto de la ejecución. No editar modelos terminados. Si cambió el experimento, iniciar otra ejecución.
- **Dataset de origen desconocido.** Añadir `data.strata`, sin inventar etiquetas de procedencia.
- **Fallo científico.** Revisar `error.txt` y `manifest.json`; un candidato fallido también conserva el detalle en `search_checkpoint.json`. El HTML no convierte fallos en resultados válidos.
- **Memoria/procesos.** Comenzar con `--smoke` y un proceso; aumentar `--workers` únicamente para entrenamiento/búsqueda. El conjunto original completo se validó con dos procesos.

El alcance inicial es potenciales de pares. Tres cuerpos, DFT nuevo, simulaciones extensas de dinámica molecular y nuevas líneas de polarización quedan para trabajo posterior.
