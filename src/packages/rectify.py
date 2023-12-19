# 2020-12-07
import numpy as np
import scipy.linalg

def rectify( F, im1, im2 ):
    """
    Simple epipolar rectification.  
    H1, H2, im1r, im2r = rectify( F, im1, im2 )
    """

    t1 = im1.dtype
    t2 = im2.dtype

    im1 = im1.astype( 'float' )
    im2 = im2.astype( 'float' )

    if len( im1.shape ) == 3:
        im1 = 0.2989 * im1[:,:,0] + 0.587 * im1[:,:,1] + 0.1140 * im1[:,:,2]
        

    if len( im2.shape ) == 3:
        im2 = 0.2989 * im2[:,:,0] + 0.587 * im2[:,:,1] + 0.1140 * im2[:,:,2]

    # to the old coordinate system
    F = F[[1,0,2]][:,[1,0,2]]
    
    sz = [ im1.shape[0], im1.shape[1] ]
    
    H1, H2, FF = rectprimitive( F, sz, sz );

    # to the new coordinate system
    H1 = H1[[1,0,2]][:,[1,0,2]]
    H2 = H2[[1,0,2]][:,[1,0,2]]

    sz = [ im1.shape[1], im1.shape[0] ]
    
    H1, H2 = rpairalign( H1, H2, sz, sz )

    im1r, im2r = rpairproj( im1, im2, H1, H2 )

    im1r = im1r.astype( t1 )
    im2r = im2r.astype( t2 )


    return H1, H2, im1r, im2r

#function [HH1, HH2, FF] = 
def rectprimitive( F, sz1, sz2 ):

    # Epipoles
    e1 = scipy.linalg.null_space( F )
    e2 = scipy.linalg.null_space( F.T )

    # Mapping the epipole to infinity ([15.5.2003/1])
    H1 = ep2inf( e1, sz1 )
    H2 = ep2inf( e2, sz2 )

    # return if ep2inf not successful
    if H1 is None or H2 is None:
        return None, None, None

    # Correction of H1 to match H2

    FF = np.linalg.inv( H2 ).T @ F @ np.linalg.inv( H1 )
    # now FF = [a 0 b; 0 0 0; c 0 d ];
    # let find rot2 (tilt), such that b = c;

    a = FF[0,0]
    b = FF[0,2]
    c = FF[2,0]
    d = FF[2,2]

    H = np.array( [ [ b, 0, -a ], [0, 0, 0], [a, 0, b ] ] / np.sqrt( a**2 + b**2 ) )

    if b < 0:
        H = -H

    H[1,1] = 1.0

    k = ( a * d - b * c ) / ( a**2 + b**2 )
    du = -( a * c + b * d ) / ( a * d - b * c )

    # mirroring detection / elimination (image is not mirrored but rotated)
    sv = 1.0 if k > 0 else -1.0
        
    H = np.array( [[k, 0, k * du], [0, sv, 0], [0, 0, 1]] ) @ H
    
    H1 = H @ H1
    FF = np.linalg.inv( H2 ).T @ F @ np.linalg.inv( H1 )
    
    return H1, H2, FF

def ep2inf( e, sz ):
    e = e.reshape(3)
    
    # shift origin to image center
    T = np.array( [ [ 1., 0., -sz[0]/2], [0., 1., -sz[1]/2], [0., 0., 1.] ] )

    # shift the epipole, normalize 3rd coordinate
    ep = T @ e

    if ep[2] < 0:
        ep = -ep

    # R:Rotation such that epipole is [e1;e2;e3] -> [0;f;e3] or [0;-f;e3]
    # (up to scale), where f = sqrt(e1^2 + e2^2). The direction of rotation is
    # determined by signum of ep(2)

    # G:Translation of epipole to inf, direction is based on signum of ep(2)

    f = np.sqrt( ep[0]**2 + ep[1]**2 )
    F = ep[2] / f
    S = 1 - F**2 * sz[1]**2 / 4

    if ep[1] >= 0:
        # epipole -> [0;f;e3]
        R = np.array( [[ ep[1], -ep[0], 0],[ep[0], ep[1], 0], [0, 0, f] ] )
        G = np.array( [[ np.sqrt(S), 0, 0],[0, S, 0],[0, -F, 1] ] )
    else:
        # epipole -> [0;-f;e3]
        R = np.array( [[ -ep[1], ep[0], 0],[-ep[0], -ep[1], 0], [0, 0, f] ] )
        G = np.array( [[ np.sqrt(S), 0, 0],[0, S, 0],[0, F, 1] ] )

    H = G @ R @ T

    #nep = H @ e
        
    return H
    
def rbb( H, sz ):
    corners1 = np.array( [ [ 0, sz[0]-1, sz[0]-1, 0 ],
                           [ 0, 0, sz[1]-1, sz[1]-1 ],
                           [ 1, 1, 1, 1 ] ] )

    corners2 = H @ corners1

    corners2 = corners2[:2] / corners2[2]


    cmin = np.floor( corners2.min( axis=1 ) )
    cmax = np.ceil( corners2.max( axis=1 ) )

    return cmin, cmax


def rpairbb( H1, H2, sz1, sz2 ):

    cmin1, cmax1 = rbb( H1, sz1 )
    cmin2, cmax2 = rbb( H2, sz2 )

    if cmin1 is None:
      return None, None, None, None


    ymin = max( cmin1[1], cmin2[1] )
    ymax = min( cmax1[1], cmax2[1] )
    
    cmin1[1] = ymin
    cmin2[1] = ymin

    cmax1[1] = ymax
    cmax2[1] = ymax

    return cmin1, cmax1, cmin2, cmax2

def rpairalign( H1, H2, sz1, sz2 ):

    cmin1, cmax1, cmin2, cmax2 = rpairbb( H1, H2, sz1, sz2)

    dy = - cmin1[1]
    dx1 = - cmin1[0]
    dx2 = - cmin2[0]

    H1 = np.array( [[1, 0, dx1],[0, 1, dy],[0, 0, 1 ]] ) @ H1
    H2 = np.array( [[1, 0, dx2],[0, 1, dy],[0, 0, 1 ]] ) @ H2

    return H1, H2                    

def rpairproj( im1, im2, H1, H2 ):
    sz1 = [ im1.shape[1], im1.shape[0] ]
    sz2 = [ im2.shape[1], im2.shape[0] ]

    cmin1, cmax1, cmin2, cmax2 = rpairbb( H1, H2, sz1, sz2 )

    rim1 = rimgproj( im1, H1, cmax1.astype( 'int' ) + 1 )
    rim2 = rimgproj( im2, H2, cmax2.astype( 'int' ) + 1 )

    return rim1, rim2


def rimgproj( im1, H, sz2 ):
    x2, y2 = np.meshgrid( np.arange( 0, sz2[0] ),
                          np.arange( 0, sz2[1] ) )

    H = np.linalg.inv( H )

    w1 = H[2,0] * x2 + H[2,1] * y2 + H[2,2]                                    
    x1 = ( H[0,0] * x2 + H[0,1] * y2 + H[0,2] ) / w1
    y1 = ( H[1,0] * x2 + H[1,1] * y2 + H[1,2] ) / w1
    
    # TODO better interpolation
    x1 = np.round( x1 ).astype( 'int' )
    y1 = np.round( y1 ).astype( 'int' )
    
    ok = ( x1 >= 0 ) * ( y1 >= 0 ) * ( x1 < im1.shape[1] ) * ( y1 < im1.shape[0] )
    
    im2 = np.zeros( (sz2[1], sz2[0]), dtype=im1.dtype )


    im2[y2[ok],x2[ok]] = im1[y1[ok],x1[ok]]
    
    return im2
    
    

#

#    TODO
#    [u1 v1] = rcoordproj( inv(H), [cmin(:) sz(:)] );
#    img2 = interp2( img1, v1, u1 );
#    img2(isnan(img2)) = 0;

#    return  img2



