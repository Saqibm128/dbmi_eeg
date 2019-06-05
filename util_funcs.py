import pickle as pkl
import json, os
from addict import Dict
from pathos.multiprocessing import Pool
import pandas as pd
import numpy as np
import pymongo

TOTAL_NUM_FILES = 2012

def get_mongo_client(path = "config.json"):
    config = read_config(path)
    if "mongo_uri" not in config.keys():
        return pymongo.MongoClient()
    else:
        mongo_uri = config["mongo_uri"]
        return pymongo.MongoClient(mongo_uri)

def read_config(path="config.json"):
    return json.load(open(path, "rb"))

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
