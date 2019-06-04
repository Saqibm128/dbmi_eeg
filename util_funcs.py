import pickle as pkl
import json, os
from addict import Dict
from pathos.multiprocessing import Pool
import pandas as pd
import numpy as np

TOTAL_NUM_FILES = 2012

def return_seizure_types():
    return ['ABSZ', 'CPSZ', 'FNSZ', 'GNSZ', 'MYSZ', 'SPSZ', 'TCSZ', 'TNSZ']

def read_config():
    return json.load(open("config.json", "rb"))

def read_preproc_1(id):
    root_path = read_config()["preprocessed_1"]
    return pkl.load(open(os.path.join(root_path, "seiz_{}.pkl".format(id)), "rb"));

def read_preproc_2(id, debug=True):
    if id % 100 == 0 and debug:
        print("Starting reading {}".format(id))
    root_path = read_config()["preprocessed_2"]
    return pkl.load(open(os.path.join(root_path, "seiz_{}.pkl".format(id)), "rb"));

def read_all(use_1=False, num_workers=4, num_files=TOTAL_NUM_FILES):
    p = Pool(processes = num_workers)
    if use_1:
        return p.map(read_preproc_1, [i for i in range(num_files)])
    else:
        return p.map(read_preproc_2, [i for i in range(num_files)])


if __name__ == "__main__":
    print(read_config())
    print(read_preproc_1(1))
    print(read_preproc_2(1))
    print(read_all_into_df(num_files=100))
