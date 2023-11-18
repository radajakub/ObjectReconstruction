import numpy as np


# transform euclidean coordinates to projective coordinates
def e2p(u_e: np.ndarray) -> np.ndarray:
    return np.vstack((u_e, np.ones(u_e.shape[-1], dtype=u_e.dtype)))


# transform projective coordinates to euclidean coordinates
def p2e(u_p: np.ndarray) -> np.ndarray:
    return u_p[:-1] / u_p[-1]


# compute length of column vectors in matrix x
def vlen(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x**2, axis=0))


# construct skew-symmetric matrix from vector x -> [x]_x
def sqc(x: np.ndarray) -> np.ndarray:
    b1, b2, b3 = x[0], x[1], x[2]
    return np.array([[0, -b3, b2], [b3, 0, -b1], [-b2, b1, 0]])


def EutoRt(E: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> list[(np.ndarray, np.ndarray)]:
    # compute SVD of E
    U, S, Vt = np.linalg.svd(E)
    V = Vt.T

    # verify digonal matrix
    if not np.isclose(S[0], S[1]) or not np.isclose(S[2], 0):
        return np.array([]), np.array([])

    # ensure rotation matrix
    U = np.linalg.det(U) * U
    V = np.linalg.det(V) * V

    for alpha in [-1, 1]:
        for beta in [-1, 1]:
            W = np.array([[0, alpha, 0], [-alpha, 0, 0], [0, 0, 1]])
            R = U @ W @ V.T
            t = (beta * U[:, 2]).reshape(3, 1)
            t = t / np.sqrt(np.sum(t ** 2))
            # verify chirality
            P1 = np.eye(3, 4)
            P2 = np.hstack((R, t))  # rotated and moved camera
            X = Pu2X(P1, P2, u1, u2)
            u1p = P1 @ X
            u2p = P2 @ X
            if np.all(u1p[2] > 0) and np.all(u2p[2] > 0):
                return R, beta * t

    return np.array([]), np.array([])


def Pu2X(P1: np.ndarray, P2: np.ndarray, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    assert (P1.shape == (3, 4))
    assert (P2.shape == (3, 4))
    assert (X1.shape[0] == 3)
    assert (X2.shape[0] == 3)

    X1 = p2e(X1)
    X2 = p2e(X2)

    p11, p12, p13 = P1
    p21, p22, p23 = P2
    X = np.zeros((4, X1.shape[1]))
    for i, (x1, x2) in enumerate(zip(X1.T, X2.T)):
        u1, v1 = x1
        u2, v2 = x2
        D = np.array([u1 * p13 - p11, v1 * p13 - p12, u2 * p23 - p21, v2 * p23 - p22])
        U, _, _ = np.linalg.svd(D.T @ D)
        X[:, i] = U[:, -1]

    return e2p(p2e(X))


def err_F_sampson(F: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    assert (F.shape == (3, 3))
    assert (u1.shape[0] == 3)
    assert (u2.shape[0] == 3)
    assert (u1.shape[1] == u2.shape[1])
    S = np.array([[1, 0, 0], [0, 1, 0]])
    SF = S @ F
    SFT = S @ F.T

    u1 = e2p(p2e(u1))
    u2 = e2p(p2e(u2))

    err = np.power(np.array([(yi.T @ F @ xi) / np.sqrt(np.sum(np.power(SF @ xi, 2)) + np.sum(np.power(SFT @ yi, 2))) for xi, yi in zip(u1.T, u2.T)]), 2)
    return err


if __name__ == '__main__':
    def test(args, result, expected, title):
        correct = np.all(np.isclose(result, expected, rtol=1e-04, atol=1e-04))
        print(f'-- test {title} {"yes" if correct else "no"}')
        if not correct:
            print(f'input: {args}')
            print(f'result: {result}')
            print(f'expected: {expected}')

    xe2p = np.array([[1, 3, 5], [2, 4, 6]])
    xe2pres = np.array([[1, 3, 5], [2, 4, 6], [1, 1, 1]])
    test(xe2p, e2p(xe2p), xe2pres, 'e2p')

    xp2e = np.array([[1, 3, 5], [2, 4, 6], [1, 1, 1]])
    xp2eres = np.array([[1, 3, 5], [2, 4, 6]])
    test(xp2e, p2e(xp2e), xp2eres, 'p2e')

    vlen_in = np.array([[0.5, 3], [0.5, 4], [1, 0]])
    vlen_out = np.array([[1.2247, 5]])
    test(vlen_in, vlen(vlen_in), vlen_out, 'vlen')

    xsqc = np.array([1, 2, 3])
    xsqc_res = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    test(xsqc, sqc(xsqc), xsqc_res, 'sqc')

    # Pu2X test
    P1 = np.array([[1.5e+03, 0.0e+00, 5.0e+02, 0.0e+00],
                   [0.0e+00, 1.5e+03, 4.0e+02, 0.0e+00],
                   [0.0e+00, 0.0e+00, 1.0e+00, 0.0e+00]])
    P2 = np.array([[1.56558935e+03,  3.32824903e+02,  8.29552630e+02,  1.55563492e+03],
                   [-5.71616762e+02,  1.43790193e+03,  8.63534776e+02,  1.01015254e+03],
                   [-7.19612060e-02, -2.59280323e-01,  9.63117490e-01,  3.03045763e-01]])
    u1 = np.array([[4.90783000e+03, 2.90121485e+03, 2.75762736e+03, 4.13630507e+03,
                    5.10221938e+03, 5.31073353e+03, 4.56859799e+03, 4.06704528e+03,
                    5.34835581e+03, 3.74453587e+03],
                   [3.21151992e+03, 2.46881780e+03, 4.81271884e+03, 3.45359403e+03,
                    5.28770272e+03, 2.55550795e+03, 3.44014006e+03, 4.49850055e+03,
                    3.62952335e+03, 3.30636198e+03],
                   [5.49250632e+00, 5.29449686e+00, 5.58559321e+00, 5.04672310e+00,
                    5.87913324e+00, 5.82965907e+00, 5.62645130e+00, 5.86568301e+00,
                    5.25978304e+00, 5.64966422e+00]])
    u2 = np.array([[8.59315687e+03, 6.29065538e+03, 6.70785818e+03, 7.74289797e+03,
                    9.32711663e+03, 8.92284109e+03, 8.31367578e+03, 8.06584844e+03,
                    9.07394073e+03, 7.42915685e+03],
                   [5.90191140e+03, 5.82184105e+03, 8.26758571e+03, 6.12544729e+03,
                    8.03356587e+03, 5.28701982e+03, 6.32342151e+03, 7.65398987e+03,
                    5.91470051e+03, 6.52523114e+03],
                   [5.31391195e+00, 5.32940958e+00, 5.17776678e+00, 4.92885749e+00,
                    5.32495367e+00, 5.71581637e+00, 5.45207745e+00, 5.53711944e+00,
                    4.96187308e+00, 5.50585528e+00]])
    res = np.array([[1.44105122,  0.16931095, -0.02508809, 1.26526905, 1.40484962, 1.56198789,
                     1.22266428, 0.8029178, 1.85894671,  0.60946552],
                    [0.67634492, 0.2340127, 1.62829124,  1.1117024, 1.89879361, 0.1368106,
                     0.82244337, 1.49402216, 1.02895271,  0.69026395],
                    [5.49250632, 5.29449686, 5.27867346,  5.90193675, 5.71409738, 5.66063233,
                     5.85943931, 6.15063049, 5.36401246,  5.60161186]])

    test([], Pu2X(P1, P2, u1, u2), res, 'Pu2X')

    # sampson test
    F = np.array([[2.31912901e-08, -1.60937201e-07,  2.98870940e-04],
                  [1.35268878e-07,  1.16310658e-07, -5.45600497e-04],
                  [-5.69944882e-04,  4.38442107e-04,  2.84464163e-01]])
    u1 = np.array([[4.90783000e+03, 2.90121485e+03, 2.75762736e+03, 4.13630507e+03, 5.10221938e+03, 5.31073353e+03, 4.56859799e+03, 4.06704528e+03, 5.34835581e+03, 3.74453587e+03],
                   [3.21151992e+03, 2.46881780e+03, 4.81271884e+03, 3.45359403e+03, 5.28770272e+03,
                       2.55550795e+03, 3.44014006e+03, 4.49850055e+03, 3.62952335e+03, 3.30636198e+03],
                   [5.49250632e+00, 5.29449686e+00, 5.58559321e+00, 5.04672310e+00, 5.87913324e+00, 5.82965907e+00, 5.62645130e+00, 5.86568301e+00, 5.25978304e+00, 5.64966422e+00]])
    u2 = np.array([[8.59315687e+03, 6.29065538e+03, 6.70785818e+03, 7.74289797e+03, 9.32711663e+03, 8.92284109e+03, 8.31367578e+03, 8.06584844e+03, 9.07394073e+03, 7.42915685e+03],
                   [5.90191140e+03, 5.82184105e+03, 8.26758571e+03, 6.12544729e+03, 8.03356587e+03,
                       5.28701982e+03, 6.32342151e+03, 7.65398987e+03, 5.91470051e+03, 6.52523114e+03],
                   [5.31391195e+00, 5.32940958e+00, 5.17776678e+00, 4.92885749e+00, 5.32495367e+00, 5.71581637e+00, 5.45207745e+00, 5.53711944e+00, 4.96187308e+00, 5.50585528e+00]])
    res = np.array([1.17767014e-25, 3.65496535e-26, 3.73504673e+00, 1.36240281e+01, 4.01348396e+00,
                   2.44827498e+01, 3.49805517e+00, 2.55861025e+01, 3.00939701e+01, 6.12625549e-01])
    test([], err_F_sampson(F, u1, u2), res, 'sampson')
