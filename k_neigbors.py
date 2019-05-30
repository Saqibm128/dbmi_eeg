from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer
import util_funcs
import pandas as pd
import numpy as np

from sacred import Experiment
ex = Experiment('k_neighbors_eeg')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create())

@ex.capture
def generate_x_y(use_1, num_files, num_workers=6):
    data_all = util_funcs.read_all(use_1, num_workers, num_files)
    x_data = np.vstack([instance.data.mean(axis=0) for instance in data_all])
    y_data_strings = [instance.seizure_type for instance in data_all]
    y_data = pd.DataFrame(
        index=range(num_files),
        columns=set(y_data_strings)
        ).fillna(0)
    for i in range(num_files):
        y_data.loc[i, y_data_strings[i]] = 1
    return x_data, y_data

@ex.named_config
def debug_config():
    num_files = 200

@ex.config
def config():
    use_1 = False
    num_files = util_funcs.TOTAL_NUM_FILES
    parameters = {
        'knn__n_neighbors': [1,2,4,8,16],
        'knn__p': [1,2]
    }

@ex.automain
def run(parameters, num_files, use_1):
    steps = [
        # ('feat_sel', SelectPercentile(chi2, 20)),
        # ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
        ]

    pipeline = Pipeline(steps)

    gridsearch = GridSearchCV(pipeline, parameters, cv=5, scoring = make_scorer(f1_score, average="weighted"))

    x_data, y_data = generate_x_y(use_1, num_files)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    gridsearch = gridsearch.fit(X_train, y_train)
    best_pipeline = gridsearch.best_estimator_

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    f1_res = f1_score(y_pred, y_test, average="weighted")



    print("Best F1 Score: {}".format(f1_res))

    return f1_res, gridsearch.cv_results_
