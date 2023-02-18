
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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