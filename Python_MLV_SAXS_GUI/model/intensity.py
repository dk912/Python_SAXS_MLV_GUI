import numpy as np
from model.form_factor import form_factor
from model.structure_factor import structure_factor

def intensity(parm, Qz):
    A = parm[9]
    Ndiff = parm[7]

    ff = form_factor(parm, Qz)
    sf = structure_factor(parm, Qz)

    I = ((ff**2) * sf + Ndiff * (ff**2)) * (Qz**-2)
    I = I / I.max()
    I = I + A
    return I / I.max()
