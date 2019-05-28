import pickle as pkl
import json, os

def read_config():
    return json.load(open("config.json", "rb"))

def read_preproc_1(id):
    root_path = read_config()["preprocessed_1"]
    return pkl.load(open(os.path.join(root_path, "seiz_{}.pkl".format(id)), "rb"));

def read_preproc_2(id):
    root_path = read_config()["preprocessed_2"]
    return pkl.load(open(os.path.join(root_path, "seiz_{}.pkl".format(id)), "rb"));


if __name__ == "__main__":
    print(read_config())
    print(read_preproc_1(1))
    print(read_preproc_2(1))
