import os
import errno
import numpy as np
import pandas as pd

from typing import List



def get_dic_list_npy(path: str, quantitys: str):
    """ Select a dict from a numpy array based on keys as quantitys

    : path : path of the dict npy file
    : quantitys : target keys
    : return : needed dict
    """
    dwarfs_dict = np.load(path).item()
    dict_need = {q: dwarfs_dict[q] for q in quantitys}
    return  dict_need


def print_sep_line():
    print('------------------------------------------------------------ \n')


def dist2(x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """ Calculate the square of distance between a numpy array and (x0, y0).

    : x : x coordinates
    : y : y coordinates
    : x0 : x center of the reference point
    : y0 : y center of the reference point

    : return : number of pixels of the inner aperture
    """
    dx = x - x0
    dy = y - y0
    return  dx ** 2 + dy ** 2


def create_dir(dir_name: str):
    """ Create directory with a name 'dir_name' """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def df_concat(paths: List[str]) -> pd.DataFrame:
    """ Concatenate multiple pandas dataframe.

    : paths : a list of paths for csv files
    : return : concatenated dataframe
    """
    dfs = [pd.read_csv(path) for path in paths]
    df_con = pd.concat(dfs)
    return  df_con
