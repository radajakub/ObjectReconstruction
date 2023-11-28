#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def vnz( x ):
    return x / np.sqrt( ( x**2 ).sum( axis=0 ) )

def sqc( x ):
    return np.array( [ [ 0, -x[2,0], x[1,0] ],
                       [ x[2,0], 0, -x[0,0] ],
                       [ -x[1,0], x[0,0], 0 ] ] )

def p2e( u ):
    return u[:-1] / u[-1]

def vlen( x ):
    return np.sqrt( ( x**2 ).sum( axis=0 ) )

def e2p( u ):
    return np.vstack( ( u, np.ones( ( 1, u.shape[1] ) ) ) )

def p3p_grunert( Xw, U ):
    """
    Three point perspective pose estim. problem - Grunert 1841.
    
    P3P_GRUNERT( Xw, u )

    solves calibrated three point
    perspective pose estimation problem by direct algorithm of Grunnert
    [Grunert-1841], as reviewed in [Haralick-IJCV1994].

    Input:
      Xw .. three 3D points in any coordinate frame (homogeneous, 4x3 matrix),
      u .. the three point projections (homogeneous, 3x3 matrix, row vectors).

    Output:
      Xc .. list, each entry containts the three 3D points in
            camera coordinate frame (homogeneous, 4x3). At most four solutions
    
    Note. For multiple real roots this algorithm does not find all solutions.
    """

    Xwe = p2e( Xw )
    a = vlen( Xwe[:,1] - Xwe[:,2] )
    b = vlen( Xwe[:,0] - Xwe[:,2] )
    c = vlen( Xwe[:,0] - Xwe[:,1] )

    if a == 0 or b == 0 or c == 0:
        print( 'There are no three unique points' )
        return []

    # ray vectors j1, j2, j3 - length is 1
    Ur = vnz( U )

    # the angles
    ca = Ur[:,1].T @ Ur[:,2]
    cb = Ur[:,0].T @ Ur[:,2]
    cg = Ur[:,0].T @ Ur[:,1]

    aa = a**2
    bb = b**2
    cc = c**2

    caca = ca**2
    cbcb = cb**2
    cgcg = cg**2

    q1 = ( aa - cc ) / bb
    q2 = ( aa + cc ) / bb
    q3 = ( bb - cc ) / bb
    q4 = ( bb - aa ) / bb

    A = [ (q1-1 )**2 - 4*cc*caca/bb,
         4*( q1*(1-q1)*cb - (1-q2)*ca*cg + 2*cc*caca*cb/bb ),
         2*( q1**2 - 1 + 2*q1**2*cbcb + 2*q3*caca - 4*q2*ca*cb*cg + 2*q4*cgcg ),
         4*( -q1*(1+q1)*cb + 2*aa*cgcg*cb/bb - (1-q2)*ca*cg ),
         (1+q1)**2 - 4*aa*cgcg/bb ]

    v = np.roots( A )

    # chose only real roots
    # TODO - numerics
    v = v[ np.abs( v.imag ) < 1e-6 * np.abs( v.real ) ].real

    # substitution for u to eq. (8), for s1 to eq. (5) and for s2, s3 to eq. (4)
    u = ( (q1-1)*v**2 - 2*q1*cb*v + 1 + q1 ) / ( 2*(cg-v*ca) )

    q = aa/(u**2+v**2-2*u*v*ca)
        #bb/(1+v**2-2*v*cb),
        #cc/(1+u**2-2*u*cg) ) )

    s1 = np.sqrt( q )
    s2 = u * s1
    s3 = v * s1

    X = [ None ] * s1.shape[0]
    for i in range( 0, s1.shape[0] ):
        X[i] = e2p( Ur * np.array( [[s1[i], s2[i], s3[i]],[s1[i], s2[i], s3[i]],[s1[i], s2[i], s3[i]]] ) )

    return X


def XX2Rt_simple( X1, X2 ):
    """
    Absolute orientation problem, simple solution.

    R, t = XX2Rt_simple( X1, X2 )
      solves absolute orientation problem using simple method.

    Input:
      X1, X2 .. 3D points (at least three) in two different Cartesian
      coordinate systems (homogeneous , 4xn matrix),

    Output:
     R .. matrix of rotation (3x3)
     t .. translation
    """

    X1 = p2e( X1 )
    X2 = p2e( X2 )

    # points relative to centroids
    c1 = X1.mean( axis=1 ).reshape((3,1))
    c2 = X2.mean( axis=1 ).reshape((3,1))

    X1 = X1 - c1
    X2 = X2 - c2

    # ROTATION

    # normal of triangle plane
    n1 = vnz( sqc(X1[:,[0]]) @ X1[:,[2]] )
    n2 = vnz( sqc(X2[:,[0]]) @ X2[:,[2]] )

    # axis and angle of rotation 1
    a = vnz( sqc(n1) @ n2 )
    c_alpha = ( n1.T @ n2 )[0,0]
    s_alpha = np.sqrt(1-c_alpha**2)

    # Rodrigues' rotation formula
    A = sqc( a )
    R1 = np.eye(3) + A * s_alpha + A@A * ( 1 - c_alpha )

    X1a = R1 @ X1;

    # rotation 2 - axis (n2) and angle
    c_alpha = ( vnz(X2) * vnz(X1a) ).sum(axis=0)
    c_alpha = c_alpha[0]
    s_alpha = np.sqrt(1-c_alpha**2)

    # Rodrigues' rotation formula
    A = sqc( n2 );
    R2 = np.eye(3) + A * s_alpha + A@A * ( 1 - c_alpha )

    # which direction?
    e_plus = ( vlen( X2 - R2 @ X1a ) ).sum()
    e_minus = ( vlen( X2 - R2.T @ X1a ) ).sum()

    if e_plus < e_minus:
        R = R2 @ R1
    else:
        R = R2.T @ R1

    # SHIFT
    t = c2 - R @ c1

    return R, t
