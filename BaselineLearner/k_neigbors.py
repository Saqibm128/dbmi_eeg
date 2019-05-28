from sklearn import pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.neighbors import KNeighborsClassifier

steps = [
    ('feat_sel', SelectPercentile(chi2, 20)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
    ]

pipeline = Pipeline(steps)
