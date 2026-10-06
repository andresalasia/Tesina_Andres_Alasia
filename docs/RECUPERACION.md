# Recuperación de referencias de investigación

Gmail se usa solo para buscar y leer; la recuperación no envía ni modifica mensajes. Buscar primero intercambios de 2022–2024 sobre tesina, BaTiO₃/BTO, FORTRAN, potenciales y LAMMPS. Ampliar fechas para fuentes `.f`, `.for`, `.f90`, `.f95` y comprimidos; revisar el contexto antes de ejecutar.

Guardar originales en subdirectorios de `private/gmail/`, preservando cada versión. El inventario privado debe contener fecha, mensaje de procedencia, nombre, tamaño, MIME, estado de recuperación y SHA-256. Comparar hashes antes de declarar duplicados: nombre y tamaño iguales no bastan. Conservar los enlaces y destinos concretos de descargas manuales en `private/gmail/README.md`. Si el conector declara el MIME no soportado, pedir la descarga manual de ese adjunto.

## Estado inicial

| Función | Referencia encontrada | Qué falta |
|---|---|---|
| Vecinos y pesos para polarización | Fuente FORTRAN, CONFIG y tabla de referencia; recuperados y contrastados | Revisar adaptación a celdas distorsionadas antes de generalizar |
| Polarización física | Resultados y tablas adicionales localizados | Descargar los adjuntos pendientes; recuperar cargas y programa que calcula P |
| Potenciales y validación de fases | Varias tablas históricas localizadas | Descargar versiones, identificar modelos M1/M2/M3 y asociar inputs/resultados |
| Energía–volumen/bulk | CLI Python reproduce M2; un archivo adicional E vs V recuperado requiere identificar sistema y unidades | Fuente FORTRAN y correspondencia inequívoca del input para contraste |
| Suavizado de M3 | No se identificó una fuente inequívoca | Recuperar algoritmo, parámetros y modelo correspondiente |
| Trayectorias LAMMPS | Contexto de simulaciones identificado | Inputs, trayectorias y metadatos de procedencia para comprobar independencia |

## Ejecutar una referencia con seguridad reproducible

1. Revisar entradas, salidas y cualquier llamada externa de la fuente; no ejecutar adjuntos indiscriminadamente.
2. Crear un directorio aislado bajo `private/`, copiar solo las entradas y compilar con diagnósticos y controles de límites, por ejemplo `gfortran -O0 -g -Wall -Wextra -fcheck=all`.
3. Guardar versión del compilador, comando, stdin, stdout/stderr, hashes y criterios de comparación. No escribir junto a los originales: algunos programas sobrescriben su tabla de salida.
4. Comenzar con un caso pequeño y contrastar la referencia antes de traducir o extender código. Mantener las convenciones y unidades explícitas.
5. Elegir con el autor qué fuentes corresponde incorporar antes de publicarlas. La PR de la CLI no incluye correos ni adjuntos.

El programa de vecinos recuperado supone centros identificados por especie, ocho vecinos Ba y seis O orientados, y radios fijos expresados mediante distancias al cuadrado. La revisión detectó casos no validados: menos centros que los solicitados, vecinos faltantes/extra, celda singular y lectura de CONFIG con velocidades sin fuerzas. La entrada recuperada usa solo posiciones y pasó la comparación; eso no valida esos casos adicionales.

GNU Fortran es una herramienta independiente, instalada en esta Mac para el contraste. No se añade a Poetry ni es requisito para ejecutar la CLI.
