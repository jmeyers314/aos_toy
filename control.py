# Separate estimation and control?


import numpy as np
import galsim


class LstSqEstimator:  # AKA pseudoinverse
    """
    Parameters
    ----------
    dz0 : array
        Intrinsic Double Zernikes.
    A : array
        Sensitivity matrix in double zernike coefficients
    """
    def __init__(self, dz0, A):
        self.dz0 = dz0
        self._DZ0 = galsim.zernike.DoubleZernike(
            self.dz0,
            # hard-coding Rubin annuli for now
            uv_inner=0.0, uv_outer=1.75,
            xy_inner=0.612*4.18, xy_outer=4.18
        )
        self.A = A

    def estimate(self, y, rtp, fields):
        """ Estimate state from measurements

        Parameters
        ----------
        y : array (nfield, jmax)
            Measurements
        rtp : float
            Rotation angle of camera in radians
        fields : list of (float, float)
            Field angle where measurements are performed in degrees.

        Returns
        -------
        x_hat : array (ndof)
            Estimated telescope state
        """
        field_x, field_y = zip(*fields)

        # Subtract intrinsic Zernikes from measurement
        dz = self._DZ0.rotate(theta_uv=rtp)
        y_intrinsic = np.array([zk.coef for zk in dz(field_x, field_y)])
        dy = y - y_intrinsic  # Don't modify y in place
        dy = dy[..., 4:]  # Exclude zero, piston, tip, tilt

        # Form the single Zernike sensitivity matrix at the rotated field angles
        A_corners = np.array([
            np.array([
                zk.coef
                for zk in galsim.zernike.DoubleZernike(
                    Ai,
                    # Rubin annuli
                    uv_inner=0.0, uv_outer=1.75,
                    xy_inner=0.612*4.18, xy_outer=4.18
                ).rotate(theta_uv=rtp)(field_x, field_y)
            ])
            for Ai in self.A
        ])
        A_corners = A_corners[..., 4:]  # Exclude zero, piston, tip, tilt
        A_corners = A_corners.reshape(A_corners.shape[0], -1).T
        dy = dy.reshape(-1)

        xhat, res, rank, s = np.linalg.lstsq(
            A_corners, dy, rcond=None
        )
        return xhat


class DirectController:
    """
    Parameters
    ----------
    estimator : object
        Estimator object with estimate method
    gain : float
        Gain on the control signal
    """
    def __init__(self, estimator, gain=1.0):
        self.estimator = estimator
        self.gain = gain

    def control(self, y, rtp, field):
        """
        Parameters
        ----------
        y : array (nfield, jmax)
            Measurements
        rtp : float
            Rotation angle of camera in radians
        field : list of (float, float)
            Field angle where measurements are performed in degrees.

        Returns
        -------
        x_hat : array (ndof)
            Estimated telescope state
        x : array (ndof)
            Correction to send.
        """
        x_hat = self.estimator.estimate(y, rtp, field)
        dx = -self.gain * x_hat
        return x_hat, dx


class PIDController:
    def __init__(self, estimator, Kp=0.3, Ki=0.3, Kd=0.3):
        self.estimator = estimator
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last = 0.0
        self.integral = 0.0

    def control(self, y, rtp, field):
        x_hat = self.estimator.estimate(y, rtp, field)

        p_term = self.Kp * x_hat

        self.integral += x_hat
        i_term = self.Ki * self.integral

        der = x_hat - self.last
        d_term = self.Kd * der

        self.last = x_hat

        return x_hat, -(p_term + i_term + d_term)


class AngeliEstimator:
    """
    Parameters
    ----------
    dz0 : array
        Intrinsic Double Zernikes.
    A : array
        Sensitivity matrix in double zernike coefficients
    X : array
        State covariance matrix
    W : array
        Measurement covariance matrix
    """
    def __init__(self, dz0, A, X, W):
        self.dz0 = dz0
        self._DZ0 = galsim.zernike.DoubleZernike(
            self.dz0,
            # hard-coding Rubin annuli for now
            uv_inner=0.0, uv_outer=1.75,
            xy_inner=0.612*4.18, xy_outer=4.18
        )
        self.A = A
        self.X = X
        self.W = W

    def estimate(self, y, rtp, fields):
        """ Estimate state from measurements

        Parameters
        ----------
        y : array (nfield, jmax)
            Measurements
        rtp : float
            Rotation angle of camera in radians
        fields : list of (float, float)
            Field angle where measurements are performed in degrees.

        Returns
        -------
        x_hat : array (ndof)
            Estimated telescope state
        """
        field_x, field_y = zip(*fields)

        # Subtract intrinsic Zernikes from measurement
        dz = self._DZ0.rotate(theta_uv=rtp)
        y_intrinsic = np.array([zk.coef for zk in dz(field_x, field_y)])
        dy = y - y_intrinsic  # Don't modify y in place
        dy = dy[..., 4:]  # Exclude zero, piston, tip, tilt

        # Form the single Zernike sensitivity matrix at the rotated field angles
        A_corners = np.array([
            np.array([
                zk.coef
                for zk in galsim.zernike.DoubleZernike(
                    Ai,
                    # Rubin annuli
                    uv_inner=0.0, uv_outer=1.75,
                    xy_inner=0.612*4.18, xy_outer=4.18
                ).rotate(theta_uv=rtp)(field_x, field_y)
            ])
            for Ai in self.A
        ])
        A_corners = A_corners[..., 4:]  # Exclude zero, piston, tip, tilt
        A_corners = A_corners.reshape(A_corners.shape[0], -1).T
        dy = dy.reshape(-1)

        # Implement Angeli equation directly
        Ainv = np.linalg.pinv(A_corners)
        W = self.W[:,4:,:,4:].reshape(len(dy), len(dy))
        tmp = Ainv @ W
        xhat = np.linalg.inv(tmp @ A_corners + self.X) @ tmp @ dy
        return xhat

