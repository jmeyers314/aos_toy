import numpy as np
import galsim


class MeasSim:
    """Simulate a measurement.

    Parameters
    ----------
    dz0 : array (kmax, jmax)
        Intrinsic Double Zernikes.
    A : array (ndof, kmax, jmax)
        Sensitivity matrix in double zernike coefficients
    fields : list of (float, float)
        Field angles where measurements are performed in degrees.
    W : array (nfield*jmax, nfield*jmax)
        Measurement noise covariance matrix

    Notes
    -----
    Specifying the fields angles here so we can pass in a single covariance
    matrix here too.  Specifying the field angles in the measurement method
    would require passing in a covariance matrix for each measurement, which we
    don't currently have a model for.
    """
    def __init__(self, dz0, A, fields, W, rng=None):
        self.dz0 = dz0
        self.A = A
        self.fields = fields
        self.W = W
        self._field_x, self._field_y = zip(*fields)
        self._DZ0 = galsim.zernike.DoubleZernike(
            self.dz0,
            # hard-coding Rubin annuli for now
            uv_inner=0.0, uv_outer=1.75,
            xy_inner=0.612*4.18, xy_outer=4.18
        )
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def measurement(self, x, rtp):
        """
        Parameters
        ----------
        x : array (ndof)
            Telescope state
        rtp : float
            Rotation angle of camera in radians

        Returns
        -------
        y : array (nfield, jmax-3)
            Noiseless measurement vector in single zernike coefficients in
            meters.
        y_noise : array (nfield, jmax-3)
            Measurement vector in single zernike coefficients in meters.
        """

        # Assemble state Double Zernike
        dz = self._DZ0 + galsim.zernike.DoubleZernike(
            np.sum(x[:, None, None] * self.A, axis=0),
            # Rubin annuli
            uv_inner=0.0, uv_outer=1.75,
            xy_inner=0.612*4.18, xy_outer=4.18
        )
        dz = dz.rotate(theta_uv=rtp)
        y = np.array([
            zk.coef
            for zk in dz(self._field_x, self._field_y)
        ])
        noise = self.rng.multivariate_normal(
            np.zeros_like(y.ravel()),
            self.W
        ).reshape(y.shape)

        # Zero out piston, tip, and tilt terms
        y[:, :4] = 0.0
        noise[:, :4] = 0.0

        return y, y + noise
