# Validación de la CLI

Verificación local del 6 de octubre de 2026. Las cifras completas y versiones están en [validation_m2.json](validation_m2.json). Se conservaron las versiones de `poetry.lock` y los archivos históricos sin cambios.

## Reproducción científica

| Cálculo | Archivado | Reajuste completo |
|---|---:|---:|
| RMSE E holdout (meV/átomo) | 3,8711425127 | 3,8711423565 |
| RMSE F holdout (meV/Å) | 134,4983451975 | 134,4983470405 |
| Red cúbica archivada (Å) | 3,9814189551 | — |
| Bulk archivado (GPa) | 137,0382653315 | — |

Todos los controles históricos cumplen las tolerancias. El reajuste se ejecutó una vez con 1020 estructuras y dos procesos, evaluando las 438 restantes. Las diferencias relativas de RMSE son aproximadamente `4,0e-8` (energía) y `1,4e-8` (fuerzas), muy inferiores al 1 %. La máxima diferencia de predicciones es `1,34e-8 eV/átomo` y `2,22e-5 eV/Å`. No se cambiaron parámetros para forzar coincidencia.

La comparación comprueba también las referencias y el orden de las estructuras. El almacenamiento de fuerzas se normalizó a estructura/átomo/xyz: los descriptores UF3 ordenan originalmente por componente dentro de cada estructura, mientras ASE devuelve átomo/xyz. Esta diferencia de almacenamiento no afecta los RMSE, pero impediría una comparación elemento a elemento sin normalizarla.

## Pruebas rápidas e integraciones

Las 31 pruebas de `unittest` pasan y cubren la carga histórica sin mutación, conservación de nodos/predicciones al guardar/cargar, fuerzas por diferencias finitas, igualdad serial/paralela y orden de muestras, fresh model y arrays de entrenamiento inmutables, y factor 2 en tablas. También cubre agrupación y estratificación, smoke posterior a partición, CLI/HTML offline, errores, hashes y reanudación de búsquedas. Se prueba la lógica de los 20 candidatos con dobles controlados, sin ajustar los 20 modelos reales.

Integraciones realizadas:

- Los seis comandos CLI, incluyendo evaluación completa, análisis reducido, entrenamiento completo histórico, exportación y fonones 3×3×3 con desplazamiento 0,05 Å.
- Cuatro candidatos reales reducidos con diferentes pesos/curvaturas, y continuación de dos a cuatro mediante `--resume`, conservando los dos modelos completos. La inversión del orden dio exactamente las mismas métricas.
- Test queda fuera de la featurización y del ajuste de la búsqueda; métricas de selección corresponden a validación. El muestreo reducido usa 12 estructuras por partición, con errores grandes que no son una medida científica de calidad.
- Verificación visual de PNG, referencias locales en HTML y separación de autovectores complejos NPZ de las bandas JSON en THz.
- `poetry check --lock`, construcción wheel/sdist e instalación del wheel local para comprobar `poetry run tesina`.

Los resultados completos de cada comando permanecen en `data/out/runs/`, excluidos de Git. La grilla real completa de 20 candidatos no se ejecutó. No se validaron una dinámica molecular extensa ni fases relajadas de menor simetría.

## Contraste FORTRAN privado

Se recuperó una fuente de vecinos/pesos y su entrada y salida de referencia. La compilación con GNU Fortran 15.2.0 y controles de límites terminó correctamente. Para 1000 centros y 5000 átomos, la salida coincide con la referencia en sus 22002 tokens, sin errores de vecinos. Un caso adicional de 27 centros y 135 átomos también coincidió en todos sus conjuntos de vecinos con un cálculo independiente de ASE. Los originales y la evidencia permanecen en `private/`.

Este control verifica la topología necesaria para polarización, no el valor de polarización ni energía–volumen/bulk: todavía faltan el programa/cargas de polarización y fuentes/inputs de esos otros cálculos. No se publicaron archivos recuperados.
