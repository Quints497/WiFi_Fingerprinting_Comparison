from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import pandas as pd
from time import time as t

# import the data
training_data = pd.read_csv("Dataset/trainingData.csv")
testing_data = pd.read_csv("Dataset/validationData.csv")
# drop columns that won't be in use
training_data = training_data.drop(columns=['SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP'])
testing_data = testing_data.drop(columns=['SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP'])

x_train = training_data.iloc[:, 0:520]
y_train_reg = training_data.iloc[:, 520:522]
y_train_class = training_data.iloc[:, 522:524]
y_train = pd.concat([y_train_reg, y_train_class], axis=1)

x_test = testing_data.iloc[:, 0:520]
y_test_reg = testing_data.iloc[:, 520:522]
y_test_class = testing_data.iloc[:, 522:524]
y_test = pd.concat([y_test_reg, y_test_class], axis=1)

x_train.replace(100, -150, inplace=True)
x_test.replace(100, -150, inplace=True)
pca = PCA(0.90)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

mse = make_scorer(mean_squared_error, greater_is_better=False)
# K-Nearest Neighbour
knnc = KNeighborsClassifier()
knnr = KNeighborsRegressor()

# Support Vector Machine
svc = SVC()
svr = SVR()

# Decision Tree
dtc = DecisionTreeClassifier()
dtr = DecisionTreeRegressor()

# Multi-layer perceptron
mlpc = MLPClassifier()
mlpr = MLPRegressor()

classification_models = (mlpc, knnc, svc, dtc)
regression_models = (mlpr, knnr, svr, dtr)

parameter_grid = {
                'mlp': {
                    'params': {
                        'hidden_layer_sizes': [(32, 32, 32), (64, 64, 64), (100, 100, 100)],
                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
                        'solver': ['lbfgs', 'sgd', 'adam'],
                        'alpha': [0.001, 0.0001, 0.00001],
                        'learning_rate': ['constant', 'invscaling', 'adaptive'],
                        'learning_rate_init': [0.001, 0.0001, 0.00001],
                    }
                },
                'knn': {
                    'params': {
                        'algorithm': ['kd_tree'],
                        'n_neighbors': [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210],
                        'leaf_size': [10, 30, 50, 70],
                        'n_jobs': [-1, 1],
                        'p': [1, 2]
                    }
                },
                'svc': {
                    'params': {
                        'C': [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210],
                        'gamma': [0.01, 0.001, 0.0001, 0.00001],
                        'kernel': ['linear'],
                        'max_iter': [1000]
                    }
                },
                'dt': {
                    'params': {
                        'splitter': ['best', 'random'],
                        'max_depth': [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210],
                        'min_samples_split': [10, 30, 50, 70, 90],
                        'min_samples_leaf': [10, 30, 50, 70, 90],
                    }
                }
}


def finding_params(model, grid, opt):
    scoring = ('accuracy' if opt == 'FLOOR' or opt == 'BUILDINGID' else mse)
    gscv = GridSearchCV(estimator=model,
                        param_grid=grid,
                        cv=5,
                        scoring=scoring,
                        verbose=3)
    gscv = gscv.fit(x_train_pca, y_train[opt])
    best_score = gscv.best_score_
    best_params = gscv.best_params_
    best_params['score'] = best_score

    return pd.DataFrame(best_params, index=[opt])


def model_details(models, location):
    t0 = t()
    floor = finding_params(models[0], parameter_grid[location]['params'], 'FLOOR')
    t1 = t()
    building = finding_params(models[0], parameter_grid[location]['params'], 'BUILDINGID')
    t2 = t()
    longitude = finding_params(models[1], parameter_grid[location]['params'], 'LONGITUDE')
    t3 = t()
    latitude = finding_params(models[1], parameter_grid[location]['params'], 'LATITUDE')
    t4 = t()
    times = [f'{t1 - t0}:.2f', f'{t2 - t1}:.2f', f'{t3 - t2}:.2f', f'{t4 - t3}:.2f']
    details = pd.concat([floor, building, longitude, latitude])
    details['time'] = times
    return details


if __name__ == "__main__":
    #m = model_details((mlpc, mlpr), 'mlp')
    k = model_details((knnc, knnr), 'knn')
    s = model_details((svc, svr), 'svc')
    d = model_details((dtc, dtr), 'dt')

    #print(m)
    print(k)
    print(s)
    print(d)

    #m.to_csv('results/mlp.csv')
    k.to_csv('results/knn.csv')
    s.to_csv('results/svm.csv')
    d.to_csv('results/dt.csv')

