"""Funciones varias usadas en selec_hyper.py"""

# Importación paquetes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import traceback
import warnings
from uuid import uuid4

from uf3.representation import bspline
from uf3.representation import process
from uf3.regression import least_squares
from uf3.util import plotting

# Paths
CURRENT_HOUR = datetime.now().strftime('%y_%m_%d_%H_%M_%S_%f')
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
DATA_IN_PATH = DATA_PATH / "in"
DATA_OUT_PATH = DATA_PATH / "out"
DATA_SELEC_HYP = DATA_OUT_PATH / CURRENT_HOUR


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
             ("O", "Ti"): 1.28,
             ("O", "Ba"): 2.04,
             ("Ba", "Ba"): 3.35,
             ("Ti", "Ba"): 2.65,
             ("Ti", "Ti"): 3.51,
            }

def get_r_max_dict(n_vecinos):
    if n_vecinos == 2:
        r_max_map_dict = {2:r_max_map_2}
    elif n_vecinos == 3:
        r_max_map_dict = {3:r_max_map_3}
    elif n_vecinos == [2,3]:
        r_max_map_dict = {2:r_max_map_2,3:r_max_map_3}
    else:
        raise ValueError("n_vecinos debe ser 2, 3 o [2, 3]; describe alcance radial, no orden de cuerpos.")
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
    for folder in ('error_e', 'error_f'):
        (DATA_SELEC_HYP / folder).mkdir(parents=True, exist_ok=True)
    plotting.density_scatter(y_e, p_e, cmap='seismic')
    plt.tight_layout()
    plt.savefig(DATA_SELEC_HYP / f'error_e/rmse_e{n_vecinos}_vecinos_kappa_{int(kappa*1000)}_{n_splines}_splines_reg1_{reg_1b:.0E}_reg2_{curv_2b:.0E}',
                bbox_inches='tight')
    plt.close()
    plotting.density_scatter(y_f, p_f, cmap='seismic')
    plt.tight_layout()
    plt.savefig(DATA_SELEC_HYP / f'error_f/rmse_f{n_vecinos}_vecinos_kappa_{int(kappa*1000)}_{n_splines}_splines_reg1_{reg_1b:.0E}_reg2_{curv_2b:.0E}',
                bbox_inches='tight')
    plt.close()

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
    (DATA_SELEC_HYP / "pot").mkdir(parents=True, exist_ok=True)
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
    """Compatibilidad con el notebook histórico, con entrenamiento aislado.

    Cada candidato usa un modelo nuevo y las mismas matrices de entrenamiento.
    Se conserva la firma y el diccionario de resultados del notebook. ``estruct``
    ya no dispara fonones al graficar: usar ``poetry run tesina phonons`` para un
    modelo seleccionado. El protocolo de esta interfaz es el holdout histórico;
    para particiones agrupadas y reanudación usar ``poetry run tesina search``.
    Los originales ``df_data``, ``training_keys`` y ``holdout_keys`` no se modifican.
    """
    from tesina.science import fit_model, save_model
    from tesina.search import _atomic_csv, _atomic_json

    global DATA_SELEC_HYP
    if set(training_keys) & set(holdout_keys):
        raise ValueError("Entrenamiento y holdout deben contener estructuras distintas.")
    DATA_SELEC_HYP = DATA_OUT_PATH / (datetime.now().strftime('%y_%m_%d_%H_%M_%S_%f') + '-' + uuid4().hex[:8])
    DATA_SELEC_HYP.mkdir(parents=True)
    n_elements = len(chemical_system.element_list)
    columns = ["N_SPLINES", "REGULARIZADOR_1B", "REGULARIZADOR_2B", "RMSE_E (meV/atom)",
               "RMSE_F (meV/angstrom)", "kappa", "cantidad_vecinos"]
    dict_resultados = {column: [] for column in columns}
    failures = []
    completed = []
    for n_vecinos, r_max_map in r_max_map_dict.items():
        for n_splines in n_splines_list:
            bspline_config = bspline.BSplineBasis(chemical_system, r_min_map=r_min_map,
                r_max_map=r_max_map, resolution_map=create_res_map(n_splines, chemical_system),
                leading_trim=leading_trim, trailing_trim=trailing_trim)
            representation = process.BasisFeaturizer(bspline_config)
            if client is None:
                df_features = representation.evaluate(df_data, energy_key=energy_key)
            else:
                df_features = representation.evaluate_parallel(df_data, client, energy_key=energy_key, n_jobs=n_batches)
            train_tuples = least_squares.dataframe_to_tuples(df_features.loc[training_keys], n_elements=n_elements, energy_key="energy")
            valid_x_e, valid_y_e, valid_x_f, valid_y_f = least_squares.dataframe_to_tuples(
                df_features.loc[holdout_keys], n_elements=n_elements, energy_key="energy")
            for reg_1b in rango_reg_1:
                for curv_2b in rango_reg_2:
                    for kappa in pesos_fuerza_energia:
                        parameters = {"N_SPLINES": n_splines, "REGULARIZADOR_1B": reg_1b,
                                      "REGULARIZADOR_2B": curv_2b, "kappa": kappa, "cantidad_vecinos": n_vecinos}
                        try:
                            model = fit_model(bspline_config, train_tuples,
                                {"ridge_1b": reg_1b, "ridge_2b": 0, "curvature_2b": curv_2b, "weight": kappa},
                                repulsion_bounds=r_min_obs)
                            predicted_e = model.predict(valid_x_e)
                            predicted_f = model.predict(valid_x_f)
                            row = {**parameters,
                                   "RMSE_E (meV/atom)": float(np.sqrt(np.mean((valid_y_e - predicted_e)**2)) * 1000),
                                   "RMSE_F (meV/angstrom)": float(np.sqrt(np.mean((valid_y_f - predicted_f)**2)) * 1000)}
                            if not np.isfinite([row["RMSE_E (meV/atom)"], row["RMSE_F (meV/angstrom)"]]).all():
                                raise ValueError("El ajuste produjo métricas no finitas.")
                            model_dir = DATA_SELEC_HYP / 'models'
                            model_dir.mkdir(exist_ok=True)
                            save_model(model, model_dir / f'model_{len(completed):04d}.json')
                        except Exception as error:
                            failures.append({**parameters, "error_type": type(error).__name__,
                                             "error": str(error), "traceback": traceback.format_exc()})
                            warnings.warn(f"Falló candidato {parameters}: {type(error).__name__}: {error}", RuntimeWarning)
                        else:
                            completed.append(row)
                            for column in columns:
                                dict_resultados[column].append(row[column])
                            # Persistir antes de gráficos: una interrupción conserva el ajuste.
                            _atomic_csv(DATA_SELEC_HYP / 'resultados.csv', completed, columns)
                            _atomic_json(DATA_SELEC_HYP / 'failures.json', failures)
                            if plot:
                                plot_err(y_e=valid_y_e, p_e=predicted_e, y_f=valid_y_f, p_f=predicted_f,
                                         kappa=kappa, n_vecinos=n_vecinos, n_splines=n_splines,
                                         reg_1b=reg_1b, curv_2b=curv_2b)
                                solutions = least_squares.arrange_coefficients(model.coefficients, bspline_config)
                                plot_pot(bspline_config=bspline_config, solutions=solutions, kappa=kappa,
                                         n_vecinos=n_vecinos, n_splines=n_splines, reg_1b=reg_1b,
                                         curv_2b=curv_2b, zoom=zoom)
                                plt.close('all')
                        _atomic_csv(DATA_SELEC_HYP / 'resultados.csv', completed, columns)
                        _atomic_csv(DATA_SELEC_HYP / 'bad_models.csv', failures)
                        _atomic_json(DATA_SELEC_HYP / 'failures.json', failures)
            _atomic_csv(DATA_SELEC_HYP / f'resultados_{n_splines}_{n_vecinos}.csv', completed, columns)
    return dict_resultados
