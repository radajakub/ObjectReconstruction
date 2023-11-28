# corresp - Package for manipulating multiview pairwise correspondences.
#
# (c) 2020-11-07 Martin Matousek
# Last change: $Date$
#              $Revision$

import numpy as np

class Corresp():
    """ Class for manipulating multiview pairwise correspondences. """

    def __init__( this, n ):
        """
        Constructor.

        obj = Corresp( n )

        Initialises empty correspondence tables.

        Input:
          n  .. number of cameras (the cameras will be identified as 0..n-1)
        """

        # number of cameras        
        this.n = n

        # image-to-image correspondences
        # Correspondences between camera i1 and i2 (i1 != i2), are stored in
        # this.m[ min(i1,i2) ][ max(i1,i2) ]. I.e., the 'matrix' this.m has
        # diagonal and under-diagonal entries empty.
        this.m  = [ [ None ] * n for i in range( 0, n ) ]

        # numbers of correspondences
        this.mcount = np.zeros( ( n, n ), dtype=int )

        # scene-to-image correspondences (pairs [X_id u_id])
        this.Xu = [ None ] * n

        this.Xucount = np.zeros( n, dtype=int )

        # flags, tentative or verified
        this.Xu_verified = [ None ] * n

        for i in range( 0, n ):
            this.Xu[i] = np.zeros( (0, 2), dtype=int )
            this.Xu_verified[i] = np.zeros( 0, dtype=bool )

        # flag for each camera, true if it is selected, false otherwise
        this.camsel = np.zeros( n, dtype=bool )

        # last used xid for automatic numbering of 3Dpoints      
        this.last_xid = -1

        # working phase        
        this.state = 'init' 
        this.lastjoin = None

        this.verbose = 1


    def add_pair( this, i1, i2, m12 ):
        """
        Add pairwise correspondences.

        obj.add_pair( i1, i2, m12 )

        Input:
            i1, i2  .. camera pair

            m12     .. image-to-image point corresp. between camera i1 and i2.
                       Rows [ ..., [u1, u2], ... ], where u1 and u2 are IDs of
                       image point in the image i1 and i2, respectively.
        """

        if this.state != 'init':
            raise Exception( 'Cannot add correspondences now.' )

        if i1 == i2:
            raise Exception( 'Pairs must be between different cameras' )

        if i1 < 0 or i2 < 0 or i1 >= this.n or i2 >= this.n:
            raise Exception( 'Image indices must be in range 0..%i-1.', this.n )

        if m12.shape[1] != 2:
            raise Exception( 'Point correspondences must be in n x 2 matrix.' )

        # ensure correct order
        if i1 > i2:
            i1, i2 = i2, i1
            m12 = m12[ :, [1,0] ]

        if not this.m[i1][i2] is None:
            raise Exception( 'Pair %i-%i allready have correspondences.', i1, i2);

        this.m[i1][i2] = m12
        this.mcount[i1,i2] = m12.shape[0]

        if this.verbose > 1:
            print( '  Image-to-Image: pair %i-%i + %i = %i' %
                       ( i1, i2, m12.shape[0], this.mcount.sum() ) )

    def start( this, i1, i2, inl, xid=None ):
        """
        Select the first two cameras.

        obj.start( i1, i2, inl, xid=None )

        Input:
          i1, i2  .. camera pair

          inl     .. inliers; indices to image-to-image correspondences between
                     the two cameras.

          xid     .. IDs of 3D points, reconstructed from inliers. Must have the
                     same size as inl or None (automatically generated)
        """

        if this.state != 'init':
            raise Exception( 'Cannot run start now.' )

        if this.verbose:
            print( 'Attaching %i,%i ---------' % ( i1, i2 ) )
            print( '  Image-to-Image total: %i' % this.mcount.sum() )


        this.camsel[i1] = True
        this.camsel[i2] = True
        this.lastjoin = i2

        this.state = 'join'

        this.new_x( i1, i2, inl, xid )

        this.state = 'clear'
        this.lastjoin = None

    def new_x( this, i1, i2, inl, xid=None ):
        """
        New 3D points.

        xid = obj.new_x( this, i1, i2, inl, xid=None )

        Input:
          i1, i2  .. camera pair

          inl     .. inliers; indices to image-to-image correspondences between
                     the two cameras.

          xid     .. IDs of 3D points, reconstructed from inliers. Must have the
                     same size as inl or None (automatically generated)

        Scene-to-image correspondences given inliers and 3D poit IDs are
        established and image-to-image correspondences between i1 and i2
        are removed.
        """

        if this.state == 'join':
            this.state = 'newx'

        if this.state != 'newx':
            raise Exception( 'Bad command order: new_x can be only after ' +
                             'a join or new_x.' )

        if i1 > i2:
            i1, i2 = i2, i1

        if not ( ( this.camsel[i1] and this.lastjoin == i2 ) or
                 ( this.camsel[i2] and this.lastjoin == i1 ) ):
            raise Exception(
                'New points can be triangulated only between the latest\n' +
                'joined camera and some allready selected camera.' )

        if xid is None:
            xid = np.arange( 0, len( inl ) ) + this.last_xid + 1

        if len( inl ) != len( xid ):
            raise Exception( 'Inliers and IDs of 3D points must have ' +
                             'the same size' )

        if len( xid ) > 0:
            this.last_xid = xid.max()

        if this.verbose:
            print( 'New X %i-%i --------------' % (i1,i2) )
            Xu_cnt_0 = this.Xucount.sum()
            Xu_verified_0 = sum( [ v.sum() for v in this.Xu_verified ] )
            m_cnt_0 = this.mcount.sum()

        n_new = len(inl)

        newXu = np.vstack( ( xid, this.m[i1][i2][ inl, 0 ] ) ).T
        this.Xu[i1] = np.vstack( ( this.Xu[i1], newXu ) )
        this.Xucount[i1] += n_new;
        this.Xu_verified[i1] = np.hstack( ( this.Xu_verified[i1],
                                            np.ones( n_new, dtype=bool ) ) )

        if this.verbose > 1:
            print( '  Scene-to-Image: i%i + %i ok = %i (%i ok)' %
                       ( i1, n_new, this.Xucount.sum(),
                               sum( [ v.sum() for v in this.Xu_verified ] ) ) )

        newXu = np.vstack( ( xid, this.m[i1][i2][ inl, 1 ] ) ).T
        this.Xu[i2] = np.vstack( ( this.Xu[i2], newXu ) )
        this.Xucount[i2] += n_new;
        this.Xu_verified[i2] = np.hstack( ( this.Xu_verified[i2],
                                                np.ones( n_new, dtype=bool ) ) )

        if this.verbose > 1:
            print( '  Scene-to-Image: i%i + %i ok = %i (%i ok)' %
                       ( i2, n_new, this.Xucount.sum(),
                             sum( [ v.sum() for v in this.Xu_verified ] ) ) )

        # remove all edges between i1 and i2
        tmp = len( this.m[i1][i2] )
        this.m[i1][i2] = None
        this.mcount[i1,i2] = 0

        if this.verbose > 1:
            print( '  Image-to-Image: pair %i-%i -%i -> 0 = %i' %
                       ( i1, i2, tmp, this.mcount.sum() ) )

        # propagate image-to-scene correspondences
        this.__propagate_x( i1, xid, '4-propagate1' )
        this.__propagate_x( i2, xid, '5-propagate2' )

        if this.verbose:
            print( '  Image-to-Image total: %i -> %i' %
                       ( m_cnt_0, this.mcount.sum() ) )
            print( '  Scene-to-Image total: %i (%i ok) -> %i (%i ok)' %
                       ( Xu_cnt_0, Xu_verified_0, this.Xucount.sum(),
                       sum( [ v.sum() for v in this.Xu_verified ] ) ) )

    def verify_x( this, i, inl ):
        """
        Set unverified scene-to-image correspondences to verified.

        obj.verify_x( i, inl )

        Input:
          i       .. the camera index

          inl     .. inliers; indices to scene-to-image correspondences between
                  image points in the camera i and the 3D points. These are
                  kept and propagated. Must be indices to un-verified
                  correspondences. Other un-verified image-to-scene
                  correspondences in the camera i are deleted.
        """


        if this.state == 'join' or  this.state == 'newx':
            this.state = 'verify'

        if this.state != 'verify':
            raise Exception( 'Bad command order: verify_x can be only after ' +
                             'a join, new_x or verify_x.' )

        if not this.camsel[i]:
            raise Exception( 'Cannot verify in a non-selected camera' )

        if this.Xu_verified[i][inl].any():
            raise Exception( '(Some) inliers are allready verified' )

        if this.verbose:
            print( 'Verify X %i --------------\n' % i )

        # set the correspondences confirmed
        this.Xu_verified[i][inl] = True

        num_outl = len( this.Xu_verified[i] ) - this.Xu_verified[i].sum()

        # get IDS of 3D points that become verified
        xid = this.Xu[i][ inl, 0 ]

        # keep only verified scene-to-image correspondences
        this.Xu[i] = this.Xu[i][ this.Xu_verified[i] ]
        this.Xu_verified[i] = np.ones( len( this.Xu[i] ), dtype=bool )
        this.Xucount[i] = len( this.Xu[i] )

        if this.verbose:
            print( '  Scene-to-Image: i%i - %i tent = %i (%i ok)' %
                       ( i, num_outl, this.Xucount.sum(),
                             sum( [ v.sum() for v in this.Xu_verified ] ) ) )

        # propagate scene-to-image correspondences from this camera
        this.__propagate_x( i, xid, '3-propagate' )


    def join_camera( this, i, inl ):
        """
        Add a camera to the set of selected cameras.

        xid  = obj.join_camera( i, inl )

        Input:
          i       .. the camera index

          inl     .. inliers; indices to scene-to-image correspondences between
                     image points in the camera i and the 3D points. These are
                     kept and propagated. Other image-to-scene correspondences
                     in the camera i are deleted.

        Output:
          xid     .. identifiers of the 3D points that are kept
        """

        if this.state != 'clear':
            raise Exception( 'Bad command order: cannot join a camera now.' )

        if not this.lastjoin is None:
            raise Exception( 'The previous join was not properly finalized.' )

        if this.camsel[i] or this.Xu[i] is None:
            raise Exception( 'Cannot join non-green camera' )

        if this.Xu_verified[i].any():
            raise Exception( 'Data structures corruption' )

        if this.verbose:
            print( '\nAttaching %i ------------' % i )

            Xu_cnt_0 = this.Xucount.sum()
            Xu_verified_0 = sum( [ i.sum() for i in this.Xu_verified ] )
            m_cnt_0 = this.mcount.sum()

        this.state = 'join'

        num_outl = len( this.Xu_verified[i] ) - len( inl )

        # add this camera to the set
        this.camsel[i] = True
        this.lastjoin = i

        # keep only the selected scene-to-image correspondences
        this.Xu[i] = this.Xu[i][ inl ]
        this.Xu_verified[i] = np.ones( len( this.Xu[i] ), dtype=bool )
        this.Xucount[i] = len( this.Xu[i] )

        if this.verbose:
            print( '  Scene-to-Image: i%i - %i tent (%i->ok) = %i (%i ok)' %
                       (i, num_outl, len( inl ), this.Xucount.sum(),
                       sum( [ i.sum() for i in this.Xu_verified ] ) ) )

        # get IDS of 3D points that are kept
        xid = this.Xu[i][:,0]

        # propagate scene-to-image correspondences from this camera
        this.__propagate_x( i, xid, '3-propagate' )

        if this.verbose:
            print( '  Image-to-Image total: %i -> %i' % ( m_cnt_0,
                     this.mcount.sum() ) )
            print( '  Scene-to-Image total: %i (%i ok) -> %i (%i ok)' %
                       ( Xu_cnt_0, Xu_verified_0,
                         this.Xucount.sum(),
                         sum( [ i.sum() for i in this.Xu_verified ] ) ) )

        return xid

    def finalize_camera( this ):
        """
        Finalize a join of a camera.
        
        obj.finalize_camera()
        """

        if this.lastjoin is None:
            raise Exception(
                'There is no previously joined camera to finalize.' )

        if not this.camsel[ this.lastjoin ]:
            raise Exception( 'Internal data corrupted.' )

        this.state = 'clear'

        i = this.lastjoin

        for q in this.camsel.nonzero()[0]:

            if not this.Xu_verified[q].all():
                raise Exception(
                  'There are some unverified scene-to-camera' +
                    'correspondences in the selected set (cam %i).' % q )

            if q == i: continue

            i1, i2 = ( q, i ) if q < i else ( i, q )

            if not this.m[i1][i2] is None:
                raise Exception(
                  ( 'Found correspondences between cameras %i-%i ' +
                  'No corresspondences must remain between selected cameras.' )
                    %( i1, i2 ) )

            if not this.m[i2][i1] is None:
                raise Exception( 'Internal data corrupted.' )

        this.lastjoin = None


    def get_m( this, i1, i2 ):
        """
        Get pairwise image-to-image correspondences.

        [ m1, m2 ] = obj.get_m( i1, i2 )

        Input:
          i1, i2  .. camera pair

        Output:
          m1, m2  .. image-to-image point correspondences between camera i1
                     and i2.
                     m1 and m2 have same sizes, m1 contains indices of points in
                     the image i1 and m2 indices of corresponding points in the
                     image i2
        """

        if i1 == i2: raise Exception( 'Pairs must be between different cameras' )

        if i1 < i2:
            m1 = this.m[i1][i2][:,0]
            m2 = this.m[i1][i2][:,1]
        else:
            m1 = this.m[i2][i1][:,1]
            m2 = this.m[i2][i1][:,0]

        return m1, m2


    def get_Xu( this, i ):
        """
        Get scene-to-image correspondences.

        mX, mu, Xu_verified = obj.get_Xu( i )

        Input:
          i       .. camera ID

        Output:
          X, u    .. scene-to-image point correspondences for the camera i.
                     x is ID of a scene points and u
                     is ID of an image points in the image i.

          Xu_verified .. boolean vector, size matching to Xu. Xu_verified(j) is
                         true if the correspondence Xu(i,:) has been verified
                         (in join_camera or verify_x), false otherwise.
        """
        X = this.Xu[i][:,0]
        u = this.Xu[i][:,1]
        Xu_verified = this.Xu_verified[i]

        return X, u, Xu_verified


    def get_Xucount( this, ilist ):
        """
        Get scene-to-image correspondence counts.

        Xucount, Xu_verifiedcount = obj.get_Xucount( ilist )

        Input:
          ilist   .. list of camera IDs

        Output:
            Xucount .. list of counts of scene-to-image point
                       correspondences for every camera in the ilist.

            Xu_verifiedcount .. counts of corespondences in the confirmed
                                state.
        """

        Xucount = this.Xucount[ ilist ]
        Xu_verifiedcount = np.zeros_like( Xucount )
        for i in range( 0, len( ilist ) ):
            Xu_verifiedcount[i] = this.Xu_verified[ ilist[i] ].sum()

        return Xucount, Xu_verifiedcount


    def get_cneighbours( this, i ):
        """"
        Neighb. selected cams related by image-to-image corr.

        ilist = obj.get_cneighbours( i )

        Input:
          i       .. the camera

        Output:
          ilist   .. row vector of neighbouring cameras, that are part of the
                     cluster and are related with the camera i by tentative
                     image-to-image correspondences.
        """

        ilist = np.zeros( this.n, dtype=bool )

        for q in range( 0, i ):
            ilist[q] = not this.m[q][i] is None

        for q in range( i+1, this.n ):
            ilist[q] = not this.m[i][q] is None

        return ( ilist * this.camsel ).nonzero()[0]


    def get_green_cameras( this, what='linear' ):
        """
        Get not-selected cameras having scene-to-image cor.

        [i, n] = obj.get_green_cameras()
        [i, n] = obj.get_green_cameras( 'logical' )

        Output:
          i   .. list of IDs of the green cameras (the first synopsis) or
                 logical array with true values for the green cameras (the
                 second synopsis)

          n   .. counts of scene points every camera can correspond to. Size
                  matching to i (!!).
        """

        i = np.zeros( this.n, dtype=bool )
        n = np.zeros( this.n, dtype=int )

        for k in range( 0, this.n ):
            if not this.camsel[k] and len( this.Xu[k] ) > 0:
                i[k] = True
                n[k] = len( np.unique( this.Xu[k][:,0] ) )

        if what == 'linear':
            i = i.nonzero()[0]
            n = n[i]

        elif what == 'logical':
            pass

        else:
            raise Exception( 'Unknown value for the 2nd parameter.' );


        return i, n


    def get_selected_cameras( this, what='linear' ):
        """
        Get allready selected cameras.

        i = obj.get_selected_cameras()
        i = obj.get_selected_cameras( 'logical' )

        Output:
            i .. list of IDs of selected cameras (the first synopsis) or
                 logical array with true values for the selected cameras (the
                 second synopsis)
        """

        if what == 'logical':
            return this.camsel
        elif what == 'linear':
            return this.camsel.nonzero()[0]
        else:
            raise Exception( 'Unknown value for the 2nd parameter.' );


    def __propagate_x( this, i, xids, substate ):
        """ Propagete scene-to-image correspondences."""

        if not this.camsel[i]:
            raise Exception( 'Cannot propagate from a non-selected camera.' )

        xids = np.unique( xids )
    
        xinx,_ = Corresp.findinx( this.Xu[i][:,0], xids )

        # selected corresponding point ids in the camera i (not unique):
        i_xids = this.Xu[i][ xinx, 0 ]
        i_uids = this.Xu[i][ xinx, 1 ]

        for q in range( 0, this.n ):
            # also red must be considered!

            if q == i: continue

            if i < q:
                i1, i2 = i, q # correspondences are in m{i,q}
                ci, cq = 0, 1 # i corresponds to the first col, q to the second
            else:
                i1, i2 = q, i # correspondences are in m{q,i}
                ci, cq = 1, 0 # i corresponds to the second col, q to the first

            if not this.m[i1][i2] is None:
                inx_i, inx_iq = Corresp.findinx( i_uids, this.m[i1][i2][:,ci] )

                if len( inx_i ) > 0:
                    xid = i_xids[ inx_i ]
                    q_uid = this.m[i1][i2][ inx_iq, cq ]

                    # do not include X-u correspondences that are allready there
                    keep = np.ones( len( xid ), dtype=bool )

                    for k in range( 0, len( xid ) ):
                        for p in ( xid[k] == this.Xu[q][:,0] ).nonzero()[0]:
                            if q_uid[k] == this.Xu[q][p,1]:
                                keep[k] = False

                    new_Xu = np.vstack( ( xid[keep], q_uid[keep] ) ).T

                    # TODO this is hack - i_uid(k) can correspond to more pairs cq
                    # but only the first is used and duplicated now
                    new_Xu = np.unique( new_Xu, axis=0 )
                    
                    this.Xu[q] = np.vstack( ( this.Xu[q], new_Xu ) )
                    this.Xu_verified[q] = np.hstack( ( this.Xu_verified[q],
                                      np.zeros( len( new_Xu ), dtype=bool ) ) )

                    this.Xucount[q] = len( this.Xu[q] )
                    if this.verbose > 1:
                        print( '  Scene-to-Image: i%i + %i tent = %i (%i ok)' %
                                 ( q, len( new_Xu ), this.Xucount.sum(),
                                 sum( [ v.sum() for v in this.Xu_verified ] ) ) )

                    # remove image-to-image correspondences propagated
                    # to scene-to-image
                    keep = np.ones( len( this.m[i1][i2] ), dtype=bool )
                    keep[ inx_iq ] = False
                    this.m[i1][i2] = this.m[i1][i2][ keep ]
                    this.mcount[i1,i2] = len( this.m[i1][i2] )
                    if len( this.m[i1][i2] ) == 0:
                        this.m[i1][i2] = None

                    if this.verbose > 1:
                        print( '  Image-to-Image: pair %i-%i -%i -> %i = %i' %
                            ( i1, i2, len( inx_iq ),
                            this.mcount[i1,i2], this.mcount.sum() ) )

    @staticmethod
    def findinx( i1, i2 ):
        inx1 = np.zeros( 0, dtype=int )
        inx2 = np.zeros( 0, dtype=int )
        for i in range( 0, len( i1 ) ):
            q2 = ( i1[i] == i2 ).nonzero()[0]
            q1 = np.zeros( len( q2 ), dtype=int ) + i
            inx1 = np.hstack( ( inx1, q1 ) )
            inx2 = np.hstack( ( inx2, q2 ) )

        return inx1, inx2
