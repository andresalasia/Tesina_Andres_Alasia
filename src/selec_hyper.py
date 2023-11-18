"""Script para obtener los hiperparámetros óptimos """

# Importación de paquetes
import os
from concurrent.futures import ProcessPoolExecutor

from uf3.data import io
from uf3.data import composition

from utils import run_selec_hyper

# Seleccionar modo de corrida
modo = 'PROD' # 'TEST' o 'PROD' 

# Seleccionar cantidad de vecinos a tomar
n_vecinos = 2 # Valores permitidos: 2, 3 y [2,3]

# Parámetros gráficas
zoom = False
plot = False 

# Características del sistema: elementos y orden de la interacción (pares o trios)
element_list = ['Ti', 'Ba', 'O']
degree = 2
chemical_system = composition.ChemicalSystem(element_list=element_list, degree=degree)

# Radios de corte para el potencial, es decir, región donde se calcula el potencial
r_min_map = {("O", "O"): 0.001,
             ("O", "Ti"): 0.001,
             ("O", "Ba"): 0.001,
             ("Ba", "Ba"): 0.001,
             ("Ba", "Ti"): 0.001,
             ("Ti", "Ti"): 0.001,
            }
r_max_map_2 = {("O", "O"): 4.42,
             ("O", "Ti"): 4.0,
             ("O", "Ba"): 4.5,
             ("Ba", "Ba"): 5.2,
             ("Ba", "Ti"): 4.07,
             ("Ti", "Ti"): 5.2,
            }
r_max_map_3 = {("O", "O"): 5.24,
             ("Ti", "O"): 4.0,
             ("Ba", "O"): 6.55,
             ("Ba", "Ba"): 7.155,
             ("Ba", "Ti"): 9.065,
             ("Ti", "Ti"): 7.155,
            }

# Para la distancia de corte mayor se prueba con dos o tres vecinos
if n_vecinos == 2:
    r_max_map_dict = {2:r_max_map_2,}
elif n_vecinos == 3:
    r_max_map_dict = {3:r_max_map_3}
elif n_vecinos == [2,3]:
    r_max_map_dict = {2:r_max_map_2,
                  3:r_max_map_3}
else:
    print(f'Valor de n_vecinos: {n_vecinos}. Valores válidos: 2, 3 y [2,3]')

# Cantidad de splines a recortar en r_min y r_max, tienen que ver con la suavidad del potencial en esos r
trailing_trim = 3
leading_trim = 0

# Parámetros asociados a los recursos destinados al ajuste en paralelo
n_cores = 4
n_batches = n_cores * 16  # Granularidad añadida para más actualizaciones de la barra de progreso
client = ProcessPoolExecutor(max_workers=n_cores) # Cliente

# Lectura de datos de estructuras del material dado
root = os.getcwd()
data_filename = os.path.join(root, "Training_set.xyz")
with open(os.path.join(root, "training_idx_70%.txt"), "r") as f:
    training_idx = [int(idx) for idx in f.read().splitlines()]      #Índice de las estructuras de training

# Creación de un DataFrame con las características de las estructuras
data_coordinator = io.DataCoordinator()
data_coordinator.dataframe_from_trajectory(data_filename,
                                           prefix='dft')
energy_key = data_coordinator.energy_key # Parámetro usado en el ajuste: nombre de la columna que tiene la E en el DF
df_data = data_coordinator.consolidate()

# Seteo de índices del DataFrame que corresponden a los conjuntos de train y test
training_keys = df_data.index[training_idx] # Índices de las estructuras del conjunto train
holdout_keys = df_data.index.difference(training_keys) # Índices de las estructuras del conjunto test

# Creación estructura para calculo de dispersión de fonones
from ase import Atoms, Atom
a = 4.012
d = a/2
estruct = Atoms([Atom('Ba',(0,0,0)), Atom('Ti',(d,d,d)), Atom('O',(d,d,0)), Atom('O',(0,d,d)),Atom('O',(d,0,d))],
                cell=[a,a,a],pbc=True)

# Distancia mínima observada entre iones en el conjunto de estructuras, para cada par 
r_min_obs = {("O", "O"): 2.34,
             ("Ti", "O"): 1.28,
             ("Ba", "O"): 2.04,
             ("Ba", "Ba"): 3.35,
             ("Ba", "Ti"): 2.65,
             ("Ti", "Ti"): 3.51,
            }

# Seteo de grilla de hiperparámetros

if modo == 'PROD':
    n_splines_list = list(range(15,28)) # Números de splines a testear
    rango_reg_1 = [10**i for i in range(-9,-3)]
    rango_reg_2 = [10**i for i in range(-7,-3)]
    pesos_fuerza_energia = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
elif modo == 'TEST':
    n_splines_list = list(range(24,26)) # Números de splines a testear
    rango_reg_1 = [10**i for i in range(-6,-5)]
    rango_reg_2 = [10**i for i in range(-10,-5)]
    pesos_fuerza_energia = [0.3]
else:
    print(f'modo={modo}. Valores aceptados: "TEST" o "PROD"')

# Ejecutar selección hiperparámetros
df_final = run_selec_hyper(chemical_system,n_splines_list,rango_reg_1,rango_reg_2,pesos_fuerza_energia,r_min_map,
                    r_max_map_dict,r_min_obs,df_data,client,energy_key,n_batches,training_keys,holdout_keys,
                    estruct,leading_trim,trailing_trim,plot,zoom)