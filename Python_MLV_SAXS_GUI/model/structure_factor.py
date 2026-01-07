import numpy as np

def structure_factor(parm, Qz):
    gma = 0.5772156649
    B = np.sqrt(6.28)

    d = parm[4]
    N = int(parm[5])
    eta = parm[6]

    sf = np.zeros_like(Qz, dtype=float)

    for i, q in enumerate(Qz):
        acc = 0.0
        for k in range(1, N):
            alpha = (d / B**2)**2 * q**2 * eta
            term = ((N - k) *
                    np.cos(k * q * d) *
                    np.exp(-alpha * gma) *
                    (np.pi * k)**(-alpha))
            acc += term
        sf[i] = N + 2 * acc

    return sf
