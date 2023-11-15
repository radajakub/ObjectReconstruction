/** \file
    Five-point relative pose problem helper: the matrix 'A' (implementation)

    \author  2010-02-22, Martin Matousek, extracted from p5_matrixA_mex.
    \date    \$Date:: 2010-02-22 18:35:05 +0100 #$
    \version \$Revision: 43 $
*/

/**  Five-point relative pose problem - the matrix 'A'.
     Computes the matrix 'A' according to  [Nister-PAMI2004].

     Input consists of a span of the essential matrix. The base vectors of the
     span comes e.g. from svd solution of u2' E u1 = 0. Essential matrix E is of
     the form E = a_x * x + a_y * y + a_zz * z + w, where a_x, a_y, a_z are
     (unknown) scalars.

     Main part of the c++ code is generated using a Maple script.
*/
void p5_matrixA( const double *x, ///<The base vector of essential matrix span
                 const double *y, ///<The base vector of essential matrix span
                 const double *z, ///<The base vector of essential matrix span
                 const double *w, ///<The base vector of essential matrix span
                 
                 /** [out] The 10x20 matrix "A". Rows correspond to 10 cubic
                     constraints. Columns correspond to indeterminates in order:
                     <tt> x^3 y^3 x^2*y x*y^2 x^2*z x^2 y^2*z y^2 x*y*z x*y
                     x*z^2 x*z z y*z^2 y*z y z^3 z^2 z 1 </tt>
                 */
                 double *A ) {

#define DECL_ONE( i ) double x##i=x[i-1], y##i=y[i-1], z##i=z[i-1], w##i=w[i-1]
  
  DECL_ONE( 1 );
  DECL_ONE( 2 );
  DECL_ONE( 3 );
  DECL_ONE( 4 );
  DECL_ONE( 5 );
  DECL_ONE( 6 );
  DECL_ONE( 7 );
  DECL_ONE( 8 );
  DECL_ONE( 9 );

#include "p5_matrixA_inner.c"

}
