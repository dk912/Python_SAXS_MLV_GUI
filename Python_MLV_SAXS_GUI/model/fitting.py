import numpy as np
from scipy.optimize import least_squares
from model.intensity import intensity

def fit_model(Qz, Iexp, Ierr, p0, bounds):
    def residuals(parm):
        Imodel = intensity(parm, Qz)
        return (Iexp - Imodel) / Ierr

    res = least_squares(residuals, p0, bounds=bounds)
    return res.x, res.cost
