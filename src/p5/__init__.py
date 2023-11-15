from p5.ext import *
import numpy as np

def p5gb( u1, u2 ):
    """
    Five-point calibrated relative pose problem (Grobner basis).
    Es = p5gb( u1, u2 ) computes the esential matrices E according to 
    Nister-PAMI2004 and Stewenius-PRS2006.

    Input:
      u1, u2 ..  3x5 matrices, five corresponding points, HOMOGENEOUS coord.

    Output:
      Es .. list of possible essential matrices
    """

    q = np.vstack( ( u1[0] * u2[0], u1[1] * u2[0], u1[2] * u2[0],
                     u1[0] * u2[1], u1[1] * u2[1], u1[2] * u2[1],
                     u1[0] * u2[2], u1[1] * u2[2], u1[2] * u2[2] ) ).T

    U, S, Vt = np.linalg.svd( q )

    XYZW = Vt[5:,:]

    A = p5_matrixA( XYZW ) # in/out is transposed (numpy data are row-wise)

    A = A[[5, 9, 7, 11, 14, 17, 12, 15, 18, 19]] @ \
            np.linalg.inv( A[[0, 2, 3, 1, 4, 8, 6, 10, 13, 16]] )

    A = A.T

    M = np.zeros( (10,10) )
    M[:6] = -A[[0, 1, 2, 4, 5, 7]]

    M[6,0] = 1
    M[7,1] = 1
    M[8,3] = 1
    M[9,6] = 1

    D, V = np.linalg.eig( M )

    ok = np.imag(D) == 0.0
    V = np.real( V )

    SOLS = V[6:9,ok] / V[9:10,ok]

    SOLS = SOLS.T

    Evec = SOLS @ XYZW[:3] + XYZW[3]

    Es = [ None ] * Evec.shape[0]
    for i in range( 0, Evec.shape[0] ):
        Es[i] = Evec[i].reshape((3,3))
        Es[i] = Es[i] / np.sqrt( np.sum( Es[i]**2 ) )

    return Es
