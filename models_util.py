
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge


def getModel(model_label):
    if model_label == 'Linear':
        regressor = make_pipeline(LinearRegression())
    elif model_label == 'Ridge':
        regressor = make_pipeline(MultiOutputRegressor(Ridge(alpha=1.0)))
    elif model_label == 'RidgeCV':
        regressor = make_pipeline(MultiOutputRegressor(RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75])))
    elif model_label == 'Polynomial':
        regressor = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    elif model_label == 'MLP':
        regressor = make_pipeline(MLPRegressor(hidden_layer_sizes=(11,11,11)))
    elif model_label == 'SVR':
        regressor = make_pipeline(MultiOutputRegressor(SVR(C=0.1, gamma="auto")))
    elif model_label == 'RFR':
        regressor = make_pipeline(RandomForestRegressor(n_estimators=100))
    if model_label == 'LinearSS':
        regressor = make_pipeline(StandardScaler(), LinearRegression())
    elif model_label == 'RidgeSS':
        regressor = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0)))
    elif model_label == 'RidgeCVSS':
        regressor = make_pipeline(StandardScaler(), MultiOutputRegressor(RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75])))
    elif model_label == 'PolynomialSS':
        regressor = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())
    elif model_label == 'MLPSS':
        regressor = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(11,11,11), solver='lbfgs'))
    elif model_label == 'SVRSS':
        regressor = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=0.1, gamma="auto")))
    elif model_label == 'RFRSS':
        regressor = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))
    elif model_label == 'MLPCVSS':
        regressor = make_pipeline(StandardScaler(), MLPRegressor(activation='relu', solver='lbfgs', hidden_layer_sizes=(6,6,6), max_iter=5000))

    return regressor

def train_linear(x, y):
    regressor = LinearRegression()
    regressor.fit(x, y)
    return regressor

def train_linearRidge(x, y):
    est = linear_model.RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75])
    regressor = MultiOutputRegressor(est)
    regressor.fit(x,y)
    return regressor

def train_polynomial(x, y):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_ = poly.fit_transform(x)
    regressor = LinearRegression()
    regressor.fit(x_, y)
    return regressor, poly

def train_neuralnetwork1(x, y):
    regressor = MLPRegressor(hidden_layer_sizes=(11,11,11))
    regressor.fit(x, y)
    return regressor

def train_neuralnetwork2(x, y):
    scaler_x = StandardScaler()
    x2 = scaler_x.fit_transform(x)

    regressor = MLPRegressor(hidden_layer_sizes=(11,11,11))
    regressor.fit(x2, y)
    return regressor, scaler_x

def train_neuralnetwork3(x, y):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x2 = scaler_x.fit_transform(x)
    y2 = scaler_y.fit_transform(y)

    regressor = MLPRegressor(hidden_layer_sizes=(11,11,11))
    regressor.fit(x2, y2)
    return regressor, scaler_x, scaler_y

def train_SVR(x, y):
    model = SVR(C=0.1, gamma="auto")
    regressor = MultiOutputRegressor(model)
    regressor.fit(x,y)
    return regressor

def train_RFR(x, y):
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(x,y)
    return regressor