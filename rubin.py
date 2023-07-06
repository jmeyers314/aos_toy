from functools import cache

import numpy as np

import batoid
from batoid_rubin import LSSTBuilder


@cache
def get_fiducial(band):
    """ Get fiducial telescope and wavelength

    Parameters
    ----------
    band : {'u', 'g', 'r', 'i', 'z', 'y'}
        Bandpass

    Returns
    -------
    fiducial : batoid.Optic
        Fiducial telescope
    wavelength : float
        Wavelength in nm
    """
    fiducial = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
    wavelength = {
        "u": 365.49,
        "g": 480.03,
        "r": 622.20,
        "i": 754.06,
        "z": 869.29,
        "y": 971.01,
    }[band]
    return fiducial, wavelength


@cache
def get_dz0(band, jmax=28, kmax=15):
    """ Get intrinsic double Zernike coefficients

    Parameters
    ----------
    band : {'u', 'g', 'r', 'i', 'z', 'y'}
        Bandpass
    jmax : int
        Maximum pupil Zernike index
    kmax : int
        Maximum field Zernike index

    Returns
    -------
    dz0 : array (kmax+1, jmax+1)
        Intrinsic double Zernike coefficients in waves

    Notes
    -----
    Coefficients start at 0, not 1.  Also, the pupil indices 1, 2, 3 are
    basically unobservable, but returned anyway.  Take care in use.
    """
    fiducial, wavelength = get_fiducial(band)
    return batoid.doubleZernike(
        fiducial,
        field=np.deg2rad(1.75),
        wavelength=wavelength*1e-9,  # nm -> m
        jmax=jmax,
        kmax=kmax,
    ) * wavelength  # waves -> nm


@cache
def get_sensitivity_dz(band, dof_idx, jmax=28, kmax=15):
    """ Get sensitivity matrix as double Zernike coefficients

    Parameters
    ----------
    band : {'u', 'g', 'r', 'i', 'z', 'y'}
        Bandpass
    dof_idx : tuple (ndof)
        Indices of degrees of freedom to include in sensitivity matrix.
        For example, (0, 1, 2, 3, 4) for the M2 hexapod, or
        (5, 6, 7, 8, 9) for the camera hexapod.
    jmax : int
        Maximum pupil Zernike index
    kmax : int
        Maximum field Zernike index

    Returns
    -------
    A : array (ndof, kmax+1, jmax+1)
        Sensitivity matrix in double Zernike coefficients in meters
    """
    ndof = len(dof_idx)
    A = np.empty((ndof, kmax+1, jmax+1))
    fiducial, wavelength = get_fiducial(band)
    dz0 = get_dz0(band, jmax, kmax)

    for iarr, idof in enumerate(dof_idx):
        arr = np.zeros(50)
        arr[idof] = 10.0
        builder = LSSTBuilder(fiducial).with_aos_dof(arr)
        telescope = builder.build()
        dz1 = batoid.doubleZernike(
            telescope,
            field=np.deg2rad(1.75),
            wavelength=wavelength*1e-9,  # nm -> m
            jmax=jmax,
            kmax=kmax,
        )
        dz1 *= wavelength  # waves -> nm
        A[iarr] = (dz1-dz0) / 10.0

    return A


@cache
def get_Q(band, dof_idx, jmax=28, kmax=15):
    ...
