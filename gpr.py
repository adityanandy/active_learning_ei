import numpy as np
import sklearn.preprocessing
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from functools import partial
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import GPy


def data_scale(df_known, fnames,
               y1l='y1'):
    """
    Scales data in the provided DataFrame and returns it.

    Parameters
    ----------
    df_known : Pandas DataFrame
        The available data, containing columns for fnames and y1l.
    fnames : list of str
        The names of the features used to predict for the target property.
    y1l : str
        The name of the target property we are predicting for.

    Returns
    -------
    x_scaler : StandardScaler
        A StandardScaler object fit to the X data df_known.
    y1_mean : np.float
        The mean of the y data in df_known.
    y1_std: np.float
        The standard deviation of the y data in df_known.

    """
    y1 = df_known[y1l].values
    y1_mean, y1_std = np.mean(y1), np.std(y1)
    if len(fnames) == 1:
        X = df_known[fnames].values.reshape(-1, 1)
    else:
        X = df_known[fnames].values
    x_scaler = sklearn.preprocessing.StandardScaler()
    x_scaler.fit(X)
    return x_scaler, y1_mean, y1_std


def process_generation_2DEI(df_known, df, fnames,
                            y1l='y1', y2l='y2'):
    """
    Fits Gaussian Process models to the data.

    Parameters
    ----------
    df_known : Pandas DataFrame
        DataFrame of all known data points.
    df : Pandas DataFrame
        DataFrame of all data points.
    fnames : list of str
        The names of the features used to predict for the target property.
    y1l : str
        The name of the first target property we are predicting for.
    y2l : str
        The name of the second target property we are predicting for.

    Returns
    -------
    df : Pandas DataFrame
        DataFrame of all data points, updated to have predicted means and standard deviations of the newest Gaussian Process models.
    gp1 : GPRegression
        The Gaussian Process model that predicts for the first target property.
    gp2 : GPRegression
        The Gaussian Process model that predicts for the second target property.

    """
    x_scaler, y1_mean, y1_std, = data_scale(df_known, fnames,
                                            y1l=y1l)
    x_scaler, y2_mean, y2_std, = data_scale(df_known, fnames,
                                            y1l=y2l)
    y1 = (df_known[y1l].values - y1_mean) / y1_std
    y2 = (df_known[y2l].values - y2_mean) / y2_std
    X = x_scaler.transform(df_known[fnames].values)
    X_all = x_scaler.transform(df[fnames].values)
    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1, lengthscale=1)
    gp1 = GPy.models.GPRegression(X, y1.reshape(-1, 1), kernel) # regression
    gp1.optimize(messages=False)
    hat_y1, var_y1 = gp1.predict(X_all)
    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1, lengthscale=1)
    gp2 = GPy.models.GPRegression(X, y2.reshape(-1, 1), kernel)
    gp2.optimize(messages=False)
    hat_y2, var_y2 = gp2.predict(X_all)
    df['hat_y1'] = hat_y1.reshape(-1, ) * y1_std + y1_mean
    df['sigma_y1'] = np.sqrt(var_y1.reshape(-1, )) * y1_std
    df['hat_y2'] = hat_y2.reshape(-1, ) * y2_std + y2_mean
    df['sigma_y2'] = np.sqrt(var_y2.reshape(-1, )) * y2_std
    return df, gp1, gp2


def gp_predict(gp_model, df_known, df, fnames,
               y1l='y1'):
    """
    Makes predictions with the Gaussian Process model.

    Parameters
    ----------
    gp_model : GPRegression
        The Gaussian Process model.
    df_known : Pandas DataFrame
        DataFrame of all known data points.
    df : Pandas DataFrame
        DataFrame of all data points.
    fnames : list of str
        The names of the features used to predict for the target property.
    y1l : str
        The name of the target property we are predicting for.

    Returns
    -------
    hat_y1 : np.array
        The posterior mean of the target property.
    std_y1 : np.array
        The posterior variance of the target property.

    """
    x_scaler, y1_mean, y1_std, = data_scale(df_known, fnames, y1l=y1l)
    y1 = (df_known[y1l].values - y1_mean) / y1_std # Z-normalizing.
    X = x_scaler.transform(df_known[fnames].values) 
    X_all = x_scaler.transform(df[fnames].values) # Z-normalizing.
    hat_y1, var_y1 = gp_model.predict(X_all)
    hat_y1 = hat_y1.reshape(-1, ) * y1_std + y1_mean # Undoing the Z-normalization.
    std_y1 = np.sqrt(var_y1.reshape(-1, )) * y1_std # Undoing the Z-normalization.
    return hat_y1, std_y1
