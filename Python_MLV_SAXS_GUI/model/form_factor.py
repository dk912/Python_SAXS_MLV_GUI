import numpy as np

def form_factor(parm, Qz):
    rhoH = parm[0] - 0.334
    zh = parm[1]
    sigmaH = parm[2]
    sigmaC = parm[3]
    rhoC = parm[8] - 0.334

    Fh = np.zeros_like(Qz)
    Fc = np.zeros_like(Qz)

    for i, q in enumerate(Qz):
        Fh[i] = (2*np.sqrt(2*np.pi)*sigmaH*rhoH *
                 np.exp(-0.5*sigmaH**2*q**2) *
                 np.cos(q*zh))
        Fc[i] = (np.sqrt(2*np.pi)*sigmaC*rhoC *
                 np.exp(-0.5*sigmaC**2*q**2))

    return Fh + Fc
