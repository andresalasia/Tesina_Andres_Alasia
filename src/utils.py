"""Funciones varias usadas en selec_hyper.py"""

# Importación paquetes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from os import mkdir, path
from pyprojroot import here

from uf3.representation import bspline
from uf3.representation import process
from uf3.regression import least_squares
from uf3.util import plotting
from uf3.forcefield import calculator
from uf3.forcefield.properties import phonon

# Paths
CURRENT_HOUR = datetime.now().strftime('%y_%m_%d_%H_%M')
ROOT = here()
DATA_PATH = ROOT / "data"
DATA_IN_PATH = DATA_PATH / "in"
DATA_OUT_PATH = DATA_PATH / "out"
DATA_SELEC_HYP = DATA_OUT_PATH / CURRENT_HOUR
mkdir(DATA_SELEC_HYP)


# Params
# Lista de elementos
element_list = ['Ti', 'Ba', 'O']  
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
# Distancia mínima observada entre iones en el conjunto de estructuras, para cada par 
r_min_obs = {("O", "O"): 2.34,
             ("Ti", "O"): 1.28,
             ("Ba", "O"): 2.04,
             ("Ba", "Ba"): 3.35,
             ("Ba", "Ti"): 2.65,
             ("Ti", "Ti"): 3.51,
            }

def get_r_max_dict(n_vecinos):
    if n_vecinos == 2:
        r_max_map_dict = {2:r_max_map_2}
    elif n_vecinos == 3:
        r_max_map_dict = {3:r_max_map_3}
    elif n_vecinos == [2,3]:
        r_max_map_dict = {2:r_max_map_2,3:r_max_map_3}
    return r_max_map_dict


def create_res_map(n_splines,chemical_system):
    """
    Devuelve el mapa de resolución para cada par interacuante, esto es el número de splines (el mismo) para cada par.

        Parameters:

            n_splines (int): número de splines para las interacciones.

            chemical_system (<class 'uf3.data.composition.ChemicalSystem'>): clase que representa el sistema químico a estudiar.

        Returns:

            r_min_obs (dict): mapa de resolución con el número de splines por interacción de pares
    """
    pairs = chemical_system.get_interactions_list()[len(chemical_system.element_list):]
    res_map_dict = {}
    for pair in pairs:
        res_map_dict[pair] = n_splines
    return  res_map_dict

def plot_err(y_e,p_e,y_f,p_f,kappa,n_vecinos,n_splines,reg_1b,curv_2b):
    """Plotear gráficas de energías y fuerzas predichas vs reales
    
        Parameter:

            y_e (np.array(), dtype=float): array con energías objetivos.

            p_e (np.array(), dtype=float): array con energías predichas.

            y_f (np.array(), dtype=float): array con fuerzas objetivos.

            p_f (np.array(), dtype=float): array con fuerzas predichas.

            n_vecinos (int): número de atomos vecinos a tomar en el ajuste. (para nombre archivo).

            kappa (folat): valor del peso en el ajuste. (para nombre archivo).

            n_splines (int): número de splines a ajustar. (para nombre archivo).

            reg_1b (float): regularizador l1. (para nombre archivo).

            curv_2b (float): regularizador l2. (para nombre archivo).
    """
    plotting.density_scatter(y_e, p_e, cmap='seismic')
    plt.tight_layout()
    plt.savefig(DATA_SELEC_HYP / f'error_e/rmse_e{n_vecinos}_vecinos_kappa_{int(kappa*1000)}_{n_splines}_splines_reg1_{reg_1b:.0E}_reg2_{curv_2b:.0E}',
                bbox_inches='tight')
    plotting.density_scatter(y_f, p_f, cmap='seismic')
    plt.tight_layout()
    plt.savefig(DATA_SELEC_HYP / f'error_f/rmse_f{n_vecinos}_vecinos_kappa_{int(kappa*1000)}_{n_splines}_splines_reg1_{reg_1b:.0E}_reg2_{curv_2b:.0E}',
                bbox_inches='tight')

def plot_pot(bspline_config,solutions,kappa,n_vecinos,n_splines,reg_1b,curv_2b,zoom=False):
    """Plotear gráficas de los potenciales obtenidos
    
        Parameter:

            bspline_config (uf3.representation.bspline.BSplineBasis): objeto con la base de bsplines.

            solutions (np.array(), dtype=float): array con los coeficientes predichos.

            kappa (folat): valor del peso en el ajuste. (para nombre archivo).

            n_vecinos (int): número de atomos vecinos a tomar en el ajuste. (para nombre archivo).

            n_splines (int): número de splines a ajustar. (para nombre archivo).

            reg_1b (float): regularizador l1. (para nombre archivo).

            curv_2b (float): regularizador l2. (para nombre archivo).

            zoom (bool): hacer zoom o no en los potenciales.
    """
    if zoom == False:
        fig, axes = plt.subplots(2, 3, figsize=(15, 6), dpi=200)
        for j, interaction in enumerate(bspline_config.interactions_map[2]):
            coefficients = solutions[interaction]
            knot_sequence = bspline_config.knots_map[interaction]
            if j<3:
                plotting.visualize_splines(coefficients, 
                                    knot_sequence, 
                                    ax=axes[0][j])
                axes[0][j].set_title("-".join(interaction))
            else:
                plotting.visualize_splines(coefficients, 
                                    knot_sequence, 
                                    ax=axes[1][j-3])
                axes[1][j-3].set_title("-".join(interaction))
        fig.tight_layout()
        plt.savefig(DATA_SELEC_HYP / f'pot/potenciales{n_vecinos}_vecinos_kappa_{int(kappa*100)}_{n_splines}_splines_reg1_{reg_1b:.0E}_reg2_{curv_2b:.0E}')
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 6), dpi=200)
        for j, interaction in enumerate(bspline_config.interactions_map[2]):
            coefficients = solutions[interaction]
            knot_sequence = bspline_config.knots_map[interaction]
            if j<3:
                plotting.visualize_splines(coefficients, 
                                    knot_sequence, 
                                    ax=axes[0][j])
                axes[0][j].set_title("-".join(interaction))
                axes[0][j].set_ylim(-0.05, 0.015)
            else:
                plotting.visualize_splines(coefficients, 
                                    knot_sequence, 
                                    ax=axes[1][j-3])
                axes[1][j-3].set_title("-".join(interaction))
                axes[1][j-3].set_ylim(-0.05, 0.015)
        fig.tight_layout()
        plt.savefig(DATA_SELEC_HYP / f'pot/potenciales{n_vecinos}_vecinos_kappa_{int(kappa*100)}_{n_splines}_splines_reg1_{reg_1b:.0E}_reg2_{curv_2b:.0E}_zoom')

def run_selec_hyper(chemical_system,n_splines_list,rango_reg_1,rango_reg_2,pesos_fuerza_energia,r_min_map,
                    r_max_map_dict,r_min_obs,df_data,client,energy_key,n_batches,training_keys,holdout_keys,
                    estruct,leading_trim=0,trailing_trim=3,plot=False,zoom=False):
    """
    Se busca el conjunto óptimo de hiperparémtros por medio de una búsqueda en grilla, los hiperparámetros son el número 
    de splines, el regularizador l1 en el primer término de la expansión del potencial, el regularizador l2 en la 
    curvatura para cada interacción de pares y el peso que se le otorga a las fuerzas y a las energías en el ajuste
    del potencial. Además del diccionario devuelto se guardan gráficas de los potenciales, los valores predichos de fuerza
    y energía vs los reales en el conjunto test, la curva de dispersión de fonones y una serie de archivos .csv con los 
    valores de rmse para fuerza y energía para cada conjunto de hiperparámetros de la grilla. Si se corre completamente el
    script los .csv con un número en su nombre son irrelevantes pues su información está contenida en el restante.

        Parameters:

            chemical_system (uf3.data.composition.ChemicalSystem): clase que representa el sistema químico a 
            estudiar.

            n_splines_list (list[int]): lista con los números de splines que se quieren incluir en la grilla.

            rango_reg_1 (list[float]): lista con los valores de regularizador l1 que se quieren incluir en la grilla.

            rango_reg_2 (list[float]): lista con los valores de regularizador l2 que se quieren incluir en la grilla.

            pesos_fuerza_energia (list[float]): lista con los valores peso que se quieren incluir en la grilla.

            r_min_map (dict{tup(str):float}): diccionario que contiene la distancia mínima a evaluar para cada par.

            r_max_map_dict (dict{int:dict{tup(str):float}}): diccionario que contiene la distancia máxima a evaluar para cada par.

            r_min_obs (dict{tup(str):float}): diccionario que contiene la distancia mínima a observada en los datos 
            para cada par.

            df_data (pandas.DataFrame): tabla con los datos de las estructuras.

            client (concurrent.futures.process.ProcessPoolExecutor): cliente para la featurización en paralelo.

            n_batches (int): número de "trabajadores" en la featurización en paralelo.

            training_keys (list[int]): ínidces de las estructuras de train en df_data.

            holdout_keys (list[int]): ínidces de las estructuras de test en df_data.

            estruct (ase.atoms.Atoms): estructura de la celda para el cálculo de la dispersión de fonones.

            leading_trim (int,=0): cantidad de splines a recortar en r pequeños.

            trailing_trim (int,=3): cantidad de splines a recortar en r grandes.

            plot (bool,=False): 'True' indica plotear gráficas, 'False' no plotear.

            zoom (bool,=False): un valor 'True' indica que las gráficas se plotean con zoom en valores pequeños del potencial.

        Returns:

            dict_resultados (dict): diccionario con el rmse en fuerza y energía para cada potencial de la grilla.
    """
    n_elements = len(chemical_system.element_list) 
    dict_resultados = {"N_SPLINES": [], "REGULARIZADOR_1B": [], "REGULARIZADOR_2B": [],
                       "RMSE_E (meV/atom)":  [], "RMSE_F (meV/angstrom)":  [], "kappa": [], "cantidad_vecinos": []}
    bad_model = {"N_SPLINES": [], "REGULARIZADOR_1B": [], "REGULARIZADOR_2B": [], "kappa": [], "cantidad_vecinos": []}

    for n_vecinos, r_max_map in r_max_map_dict.items():
        for n_splines in n_splines_list:
            # Se establece el número de splines para cada iteración y se instancia el objeto que represnta la base
            resolution_map = create_res_map(n_splines,chemical_system)
            bspline_config = bspline.BSplineBasis(chemical_system,
                                                r_min_map=r_min_map,
                                                r_max_map=r_max_map,
                                                resolution_map=resolution_map,
                                                leading_trim=leading_trim,
                                                trailing_trim=trailing_trim)

            # Calcular características para el modelado
            representation = process.BasisFeaturizer(bspline_config)
            df_features = representation.evaluate_parallel(df_data,
                                                        client,
                                                        energy_key=energy_key,
                                                        n_jobs=n_batches)

            # Puesta a punto de los datos para el ajuste
            df_slice = df_features.loc[training_keys]
            x_e, y_e, x_f, y_f = least_squares.dataframe_to_tuples(df_slice,                # x: cracterísticas utilizadas para entrenar y: target
                                                                n_elements=n_elements,
                                                                energy_key="energy")
            for reg_1b in rango_reg_1:
                for curv_2b in rango_reg_2:
                    # Regularizador
                    regularizer = bspline_config.get_regularization_matrix(ridge_1b=reg_1b,
                                                                        ridge_2b=0,
                                                                        curvature_2b=curv_2b)
                    # Definición del tipo modelo
                    model = least_squares.WeightedLinearModel(bspline_config,
                                                            regularizer=regularizer)
                    for kappa in pesos_fuerza_energia:                    
                        try:
                            model.fit(x_e, y_e, x_f, y_f, weight=kappa)
                        except:
                            bad_model['cantidad_vecinos'].append(n_vecinos)
                            bad_model['kappa'].append(kappa)
                            bad_model['N_SPLINES'].append(n_splines)
                            bad_model['REGULARIZADOR_1B'].append(reg_1b)
                            bad_model['REGULARIZADOR_2B'].append(curv_2b)
                        else:
                            # Fixear repulsión a distancias cortas
                            for pair, r_min in r_min_obs.items():
                                r_target = r_min
                                model.fix_repulsion_2b(pair,r_target=r_target,min_curvature=0.0)
                            df_holdout = df_features.loc[holdout_keys]
                            x_e, y_e, x_f, y_f = least_squares.dataframe_to_tuples(df_holdout,          # x: cracterísticas de 
                                                                            n_elements=n_elements,      # las estructuras donde
                                                                            energy_key="energy")        # se predice  y: target
                            # Predicciones
                            p_e = model.predict(x_e)
                            p_f = model.predict(x_f)
                            rmse_e = np.sqrt(np.mean(np.subtract(y_e, p_e)**2))
                            rmse_f = np.sqrt(np.mean(np.subtract(y_f, p_f)**2))

                            # Agregar datos a DataFrame de resultados
                            dict_resultados['cantidad_vecinos'].append(n_vecinos)
                            dict_resultados['kappa'].append(kappa)
                            dict_resultados['N_SPLINES'].append(n_splines)
                            dict_resultados['REGULARIZADOR_1B'].append(reg_1b)
                            dict_resultados['REGULARIZADOR_2B'].append(curv_2b)
                            dict_resultados['RMSE_E (meV/atom)'].append(rmse_e*1000)
                            dict_resultados['RMSE_F (meV/angstrom)'].append(rmse_f*1000)
                            
                            # Graficas
                            if plot:
                                for folder in ['error_e', 'error_f', 'pot', 'fonones']:
                                    if not path.exists(DATA_SELEC_HYP / folder):
                                        mkdir(DATA_SELEC_HYP / folder)
                                # Plotear y guardar gráficas errores
                                plot_err(y_e,p_e,y_f,p_f,kappa,n_vecinos,n_splines,reg_1b,curv_2b)                            

                                # Plotear potenciales
                                solutions = least_squares.arrange_coefficients(model.coefficients,
                                                                        bspline_config)
                                plot_pot(bspline_config,solutions,kappa,n_splines,reg_1b,curv_2b,zoom)

                                # Dispersión de fonones
                                calc = calculator.UFCalculator(model)
                                force_constants, path_data, bands_dict = calc.get_phonon_data(atoms=estruct, n_super=3, disp=0.05)
                                fig_ax=phonon.plot_phonon_spectrum(path_data=path_data,bands_dict=bands_dict, lw=0.3)
                                plt.savefig(DATA_SELEC_HYP / f'fonones/fonones_{n_vecinos}_vecinos_kappa_{int(kappa*100)}_{n_splines}_splines_reg1_{reg_1b:.0E}_reg2_{curv_2b:.0E}')
                                plt.close('all')
            # Guardado parcial y final de la info
            dataframe_resultados_splines = pd.DataFrame(dict_resultados)
            dataframe_resultados_splines.to_csv(DATA_SELEC_HYP / f'resultados_{n_splines}_{n_vecinos}.csv')

    # Guardado de modelos no ajustados 
    df_bad = pd.DataFrame(bad_model)
    df_bad.to_csv(DATA_SELEC_HYP / 'bad_models.csv')

    return dict_resultados