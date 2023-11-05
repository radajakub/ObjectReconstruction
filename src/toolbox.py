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
    assert (x.shape == (3, 1))
    b1, b2, b3 = x[:, 0]
    return np.array([[0, -b3, b2], [b3, 0, -b1], [-b2, b1, 0]])


def EutoRt(E: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> (np.ndarray, np.ndarray):
    assert (E.shape == (3, 3))
    assert (u1.shape[0] == 3)
    assert (u2.shape[0] == 3)
    assert (u1.shape[1] == u2.shape[1])


# compute sampson error for epipolar geometry
def err_F_sampson(F: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    assert (F.shape == (3, 3))
    assert (u1.shape[0] == 3)
    assert (u2.shape[0] == 3)
    assert (u1.shape[1] == u2.shape[1])
    S = np.eye(3)
    S[2, 2] = 0
    SF = S @ F
    SFT = S @ F.T

    # compute sampson error
    return np.zeros(shape=(1, u1.shape[1]))


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

    xsqc = np.array([[1], [2], [3]])
    xsqc_res = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    test(xsqc, sqc(xsqc), xsqc_res, 'sqc')
