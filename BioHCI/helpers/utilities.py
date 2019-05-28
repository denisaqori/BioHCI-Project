import math
import time
import datetime

import numpy as np
import os
import errno
from os.path import dirname, abspath, join
from typing import List, Dict

# this function calculates timing difference to measure how long running certain parts takes
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def create_dir(root_dir_path, subdir_name_list=None):
    '''

    Args:
        root_dir_path:
        subdir_name_list:

    Returns:

    '''
    # parent directory
    root_dir = os.path.join(get_root_path("main"), root_dir_path)
    if not os.path.exists(root_dir):
        try:
            os.makedirs(root_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # create subdirectories if there is subdir_name_list passed as a parameter
    if subdir_name_list is not None:
        for subdir_path in subdir_name_list:
            subdir = os.path.join(root_dir, subdir_path)
            if not os.path.exists(subdir):
                try:
                    os.makedirs(subdir)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

    return root_dir

def get_module_path() -> str:
    """

    Returns: absolute path of this module

    """
    return dirname(abspath(__file__))

def get_root_path(val: str) -> str:
    """
    Returns absolute paths of some important root directories in the project. Assumes project structure will remain
    the same since the paths are dependent on the position of the current module with respect to the others.

    Args:
        val (str): the name of the directory

    Returns:
        path (str): the absolute path to that directory

    """
    if val.lower() == "main":
        # two directories up from the current one (utilities)
        path = dirname(dirname(get_module_path()))
    elif val.lower() == "resources":
        path = join(dirname(dirname(get_module_path())), "Resources")
    elif val.lower() == "results":
        path = join(dirname(dirname(get_module_path())), "Results")
    elif val.lower() == "src" or val.lower() == "biohci" or val.lower() == "source":
        path = dirname(get_module_path())
    elif val.lower() == "saved_models" or val.lower() == "saved models" or val.lower() == "models":
        path = join(dirname(dirname(get_module_path())), "saved_models")
    elif val.lower() == "codebooks":
        path = join(dirname(get_module_path()), "data_processing/codebooks")
    elif val.lower() == "dataset_desc":
        path = join(dirname(get_module_path()), "data_processing/keypoint_description/dataset_descriptors")
    elif val.lower() == "dataset_eval":
        path = join(dirname(get_module_path()), "data_processing/keypoint_description/dataset_evals")
    else:
        path = None
        print("Root path for " + val + " is set to None...")
    return path

def get_files_in_dir(root_dir_path):
    if root_dir_path is not None:

        img_list = []
        for dirName, subdirList, fileList in os.walk(root_dir_path):
            for fname in fileList:
                fpath = os.path.abspath(os.path.join(dirName, fname))
                img_list.append(fpath)
        return img_list
    else:
        return None

def __map_categories(all_categories: List[str]) -> Dict[str, int]:
    """
        Maps categories from a string element to an integer.

    Args:
        categories (list): List of unique string category names

    Returns:
        cat (dict): a dictionary mapping a sting to an integer

    """
    # assert uniqueness of list elements
    assert len(all_categories) == len(set(all_categories))
    cat = {}

    all_categories.sort()
    for idx, elem in enumerate(all_categories):
        cat[elem] = idx

    return cat

def convert_categories(all_categories: List[str], categories_subset: List[str]) -> np.ndarray:
    """
    Converts a list of categories from strings to integers based on the internal attribute _cat_mapping.

    Args:
        categories (list): List of string category names of a dataset

    Returns:
        converted_categories (list): List of the corresponding integer id of the string categories

    """
    all_cat_mapping = __map_categories(all_categories)

    converted_categories = []
    for idx, elem in enumerate(categories_subset):
        assert elem in all_cat_mapping.keys()
        converted_categories.append(all_cat_mapping[elem])

    converted_categories = np.array(converted_categories)
    return converted_categories

if __name__ == "__main__":
    res = get_root_path("dataset_desc")
    print(res)