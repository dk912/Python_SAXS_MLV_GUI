import numpy as np

def electron_density(parm, dz=0.01):
    rhoH = parm[0] - 0.334
    zh = parm[1]
    sigmaH = parm[2]
    sigmaC = parm[3]
    rhoC = parm[8] - 0.334
    d = parm[4]

    z = np.arange(-d/2, d/2, dz)

    rho = (0.334 +
           rhoH*np.exp(-(z-zh)**2/(2*sigmaH**2)) +
           rhoH*np.exp(-(z+zh)**2/(2*sigmaH**2)) +
           rhoC*np.exp(-(z)**2/(2*sigmaC**2)))

    return z, rho
