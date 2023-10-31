import numpy as np


# transform euclidean coordinates to projective coordinates
def e2p(u_e):
    return np.vstack((u_e, np.ones(u_e.shape[-1], dtype=u_e.dtype)))


# transform projective coordinates to euclidean coordinates
def p2e(u_p):
    return u_p[:-1] / u_p[-1]
