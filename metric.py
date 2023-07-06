import numpy as np
import galsim


class Metrics:
    """
    Parameters
    ----------
    DZ0 : array shape (kmax, jmax)
        Double Zernike coefficients of fiducial telescope.
        Units are nanometers
    A : array shape (ndof, kmax, jmax)
        Sensitivity matrix in nanometers per (micron OR arcsec)
    field : float
        Field of view radius is degrees
    R_outer : float
        Pupil annulus outer radius in meters
    R_inner : float
        Pupil annulus inner radius in meters
    """
    def __init__(self, DZ0, A, field=1.75, R_outer=4.18, R_inner=4.18*0.612):
        self.DZ0 = DZ0
        self.A = A
        self.field = field
        self.R_outer = R_outer
        self.R_inner = R_inner

    # TODO
    # def Ixx
    # def Iyy
    # def Ixy
    # def zeta1  (unnormalized ellipticity)
    # def zeta2  (unnormalized ellipticity)
    # def third moments?

    def dz(self, x):
        """ Compute wavefront double Zernike series for given state

        Parameters
        ----------
        x : array (ndof)
            Telescope state

        Returns
        -------
        dz : galsim.zernike.DoubleZernike
            DZ series of wavefront in meters
        """
        A1 = np.einsum("a,abc->bc", x, self.A)
        coef = self.DZ0 + A1
        return galsim.zernike.DoubleZernike(
            coef,
            uv_inner=0.0, uv_outer=self.field,
            xy_inner=self.R_inner, xy_outer = self.R_outer
        )  # nanometers

    def T(self, x):
        """ Compute T = Ixx + Iyy at state x via DZs

        Parameters
        ----------
        x : array (ndof)
            Telescope state

        Returns
        -------
        T : galsim.zernike.Zernike
            T in arcsec^2 over the field-of-view as a Zernike series.
        """
        # Just use einsum for now.  TODO: fast enough?
        dz = self.dz(x) / 1e9  # nanometers -> meters
        dWdx = dz.gradX  # meters / meter
        dWdy = dz.gradY
        dWdx2 = dWdx * dWdx  # (meters / meter)^2
        dWdy2 = dWdy * dWdy
        dWdx_field = dWdx.mean_xy
        dWdy_field = dWdy.mean_xy
        dWdx2_field = dWdx2.mean_xy
        dWdy2_field = dWdy2.mean_xy

        # Now construct the PSF size T
        T = dWdx2_field + dWdy2_field - dWdx_field*dWdx_field - dWdy_field*dWdy_field
        # to arcsec
        T *= 206265**2
        return T

    def spot_size(self, x):
        """ Compute the optics PSF size in arcseconds

        This is sqrt(<T>) where T = Ixx + Iyy and the average is over the
        field of view.

        Parameters
        ----------
        x : array (ndof)
            Telescope state

        Returns
        -------
        spot_size : float
            RMS spot size in arcsec.
        """
        return np.sqrt(self.T(x).coef[1])

    def rms(self, x):
        dz = self.dz(x)
        return np.sqrt(np.sum(np.square(dz.coef[1:, 4:])))


if __name__ == "__main__":
    import batoid
    from batoid_rubin import LSSTBuilder
    # Sensitivity matrix first.  In DZs.
    # dof in order is
    # - camera z, x, y (micron)
    # - camera rx, ry (arcsec)
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    dz0 = batoid.doubleZernike(
        fiducial, np.deg2rad(1.75), 622e-9, jmax=28, kmax=10, rings=6
    )
    # We'll include j=0,1,2,3 and k=0 terms here, even though we don't use them in the control.
    A = np.empty((5,)+dz0.shape)
    for i in range(5):
        dof = np.zeros(50)
        dof[i] = 10.0
        builder = LSSTBuilder(fiducial).with_aos_dof(dof)
        telescope = builder.build()
        dz1 = batoid.doubleZernike(
            telescope, np.deg2rad(1.75), 622e-9, jmax=28, kmax=10
        )
        A[i] = (dz1-dz0) / 10.0
    # Convert both dz0 and A from waves to m
    dz0 *= 622e-9
    A *= 622e-9

    metrics = Metrics(dz0, A)

    # Focus by moving camera dz
    def focus(x0):
        x = np.zeros(5)
        x[0] = x0
        return np.sqrt(metrics.T(x).coef[1])
    from scipy.optimize import minimize_scalar, bracket
    bkt = bracket(focus)
    res = minimize_scalar(focus, bracket=bkt[0:3])

    x = np.zeros(5)
    x[0] = res.x  # best focus position
    print(f"best focus position: {x[0] = }")
    T = metrics.T(x)
    rms = metrics.spot_size(x)

    thx = np.linspace(-1.75, 1.75, 40)
    thx, thy = np.meshgrid(thx, thx)
    thr = np.hypot(thx, thy)
    w = thr <= 1.750001
    thx = thx[w]
    thy = thy[w]
    Ts = T(thx, thy)

    import matplotlib.pyplot as plt
    def colorbar(mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar

    plt.figure()
    colorbar(plt.scatter(thx, thy, c=np.sqrt(Ts)))
    plt.title(f"{rms = :.3f} arcsec")
    plt.show()
