from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression

import util_funcs
import pandas as pd
import numpy as np

from sacred import Experiment
ex = Experiment('baseline_eeg')
from sacred.observers import MongoObserver

def generate_x_y(use_1, num_files, use_expanded_y=True, num_workers=None):
    data_all = util_funcs.read_all(use_1, num_workers, num_files)
    if not use_1:
        x_data = np.vstack([instance.data.mean(axis=0) for instance in data_all])
    else:
        x_data = np.vstack([instance.data.mean(axis=0, keepdims=True) for instance in data_all])
        x_data = x_data.reshape(x_data.shape[0], -1)
    y_data_strings = [instance.seizure_type for instance in data_all]
    if use_expanded_y:
        y_data = pd.DataFrame(
            index=range(num_files),
            columns=set(y_data_strings)
            ).fillna(0)
        for i in range(num_files):
            y_data.loc[i, y_data_strings[i]] = 1
    else:
        y_data = pd.Series(y_data_strings)
    return x_data, y_data

@ex.named_config
def debug_config():
    num_files = 150

@ex.named_config
def use_1_config():
    use_1 = True

@ex.named_config
def lr_config():
    parameters = {
        'lr__tol': [0.001, 0.0001, 0.00001],
        'lr__multi_class': ["multinomial"],
        'lr__C': [0.05, .1, .2],
        'lr__solver': ["sag"],
        'lr__max_iter': [250]
    }
    clf_name = "lr"
    clf_step = ('lr', LogisticRegression())
    use_expanded_y = False

@ex.named_config
def rf_config():
    parameters = {
        'rf__criterion': ["gini", "entropy"],
        'rf__n_estimators': [100, 200, 400, 600],
        'rf__max_features' : ['auto', 'log2', .1, .4, .8],
        'rf__max_depth' : [None, 2, 4],
        'rf__min_samples_split' : [2,8],
        'rf__n_jobs' : [6],
        'rf__min_weight_fraction_leaf' : [0, 0.2, 0.5]
    }
    clf_name = "rf"
    clf_step = ('rf', RandomForestClassifier())
    return_pipeline = False #rf is too big to store

@ex.named_config
def knn_config():
    parameters = {
        'knn__n_neighbors': [1,2,4,8,16],
        'knn__p': [1,2]
    }
    clf_name = "knn"
    clf_step = ('knn', KNeighborsClassifier())
    return_pipeline = False # KNN holds all data we trained on, can't actually store this

@ex.named_config
def use_both_config():
    use_both = True

@ex.config
def config():
    use_1 = False
    use_both = False
    num_files = util_funcs.TOTAL_NUM_FILES
    parameters = {}
    clf_step = None
    return_pipeline = True
    use_expanded_y = True
    clf_name = ""
    ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.automain
def run(parameters, num_files, use_1, clf_step, return_pipeline, use_expanded_y, use_both):
    steps = [
        ('scaler', StandardScaler()),
        clf_step
        ]

    pipeline = Pipeline(steps)

    print(parameters)

    gridsearch = GridSearchCV(pipeline, parameters, cv=5, scoring = make_scorer(f1_score, average="weighted"))

    if use_both:
        x_1_data, y_data = generate_x_y(True, num_files, use_expanded_y)
        x_2_data, y_data_dup = generate_x_y(False, num_files, use_expanded_y)
        assert (y_data == y_data_dup).all().all()
        x_data = np.hstack([x_1_data, x_2_data])
    else:
        x_data, y_data = generate_x_y(use_1, num_files, use_expanded_y)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    gridsearch = gridsearch.fit(X_train, y_train)
    best_pipeline = gridsearch.best_estimator_

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    f1_res = f1_score(y_pred, y_test, average="weighted")



    print("Best F1 Score: {}".format(f1_res))
    if not return_pipeline:
        return f1_res, gridsearch.cv_results_
    else:
        return f1_res, gridsearch
