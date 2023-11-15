import p5
import numpy as np

u1 = np.random.randn( 2, 5 ) * 10
u2 = np.random.randn( 2, 5 )

u1p = np.vstack( ( u1, np.ones( ( 1, 5 ) ) ) )
u2p = np.vstack( ( u2, np.ones( ( 1, 5 ) ) ) )

Es = p5.p5gb( u1p, u2p )

print( 'det(E)                  max alg err' )

for E in Es:
    alg_err = np.sum( u2p * ( E @ u1p ), axis=0 )
    print( '{0}  {1} '.format( np.linalg.det( E ), alg_err.max() ) )
