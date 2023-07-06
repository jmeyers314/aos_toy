import numpy as np


# Just defining the interface
class StateSim:
    """ABC for state simulators."""
    def __init__(self):
        self.xs = []  # History of states

    def _step(self):
        """Return the next state.
        """
        raise NotImplementedError

    def next_x(self, dx=None):
        """
        Get the next state and optionally apply a control.

        Parameters
        ----------
        dx : array
            Control to apply to the next state.

        Returns
        -------
        x : array
            The next state.

        Notes
        -----
        Also updates the history of states in self.xs.
        """
        self.x = np.array(self._step())  # Make a copy for safety
        if dx is not None:
            self.x += dx
        self.xs.append(np.array(self.x))
        return self.x


class DiagLoopStateSim(StateSim):
    """Random perturbations of state are independent and loop over time.

    Parameters
    ----------
    x0 : array
        Initial state.
    Xdiag : array
        Diagonal of the covariance matrix of the state perturbations.
    time_scale : float
        Time scale (in iterations) of the state perturbation correlations.
    nsteps : int
        Number of steps before perturbations repeat.
    """
    def __init__(self, x0, Xdiag, time_scale, nstep=1000, rng=None):
        super().__init__()
        self.x = x0
        self.Xdiag = Xdiag
        self.time_scale = time_scale
        self.nstep = nstep
        if rng is None:
            rng = np.random.default_rng()

        t = np.arange(-nstep/2, nstep/2)
        corr = np.exp(-0.5*t**2/time_scale**2)
        pk = np.fft.fft(np.fft.fftshift(corr))
        ak = np.sqrt(2*pk)
        phi = rng.uniform(size=(nstep, len(x0)))
        zk = ak[:, None]*np.exp(2j*np.pi*phi)
        dxs = nstep/2*np.fft.ifft(zk, axis=0).real
        measured_std = np.mean(np.std(dxs, axis=0))
        dxs *= np.array(np.sqrt(Xdiag))/measured_std
        dxs -= np.mean(dxs, axis=0)
        self.dxs = dxs
        self._idx = -1

    def _step(self):
        self._idx = (self._idx + 1) % self.nstep
        return self.x + self.dxs[self._idx]


class FullLoopStateSim(StateSim):
    """TODO: Figure this out.  Maybe an AR(1) or AR(n) process?

    Paramters
    ---------
    x0 : array
        Initial state.
    X : array
        Covariance matrix of the state perturbations.
    time_scale : float
        Time scale (in iterations) of the state perturbation correlations.
    nsteps : int
        Number of steps before perturbations repeat.
    """
    def __init__(self, x0, X, time_scale, nstep=1000, rng=None):
        raise NotImplementedError
