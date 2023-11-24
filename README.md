# tesina
(** En desarrollo**) Repositorio hecho como recopilación de los scripts desarrollados al realizar mi tesina de grado en la Licenciatura en Física.

En la raíz del proyecto se encuentran dos notebooks de ejemplo sobre el proceso de ajustar un modelo y el proceso de selección de hiperparámetros.

## crear entorno virtual para correr notebooks
Ubicados en la raíz del proyecto correr las siguientes líneas

```shell
conda create -n tesina python=3.7.15
pip install wheel
pip install numba
pip install -e .
pip install spglib
pip install seekpath
pip install "phonopy>=2.6.0"
pip install uf3
```

Luego, instalar ipykernell a la hora de correr las notebooks.