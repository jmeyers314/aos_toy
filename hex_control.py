import numpy as np

from state import DiagLoopStateSim
from measurement import MeasSim
import rubin
from control import LstSqEstimator, DirectController, PIDController, AngeliEstimator
from metric import Metrics
from tqdm import tqdm


rng = np.random.default_rng(57722)
fields = [(-1.15, -1.15), (-1.15, 1.15), (1.15, -1.15), (1.15, 1.15)]
dof_idx = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
ndof = len(dof_idx)
band = "r"
fiducial, wavelength = rubin.get_fiducial("r")
dz0 = rubin.get_dz0("r")
A = rubin.get_sensitivity_dz("r", dof_idx)
metrics = Metrics(dz0, A)

# Xdiag = np.array([10**2, 50**2, 50**2, 50**2, 50**2]*2)
Xdiag = np.array([5**2, 25**2, 25**2, 25**2, 25**2]*2)
X = np.diag(Xdiag)
x0 = rng.normal(scale=300.0, size=(ndof))
state_sim = DiagLoopStateSim(x0, Xdiag, time_scale=10, rng=rng)

W = np.diag([30.0**2]*(4*29))
W_unfold = W.reshape((4, 29, 4, 29))
measure_sim = MeasSim(dz0, A, fields, W, rng)

estimator = LstSqEstimator(dz0, A)
# estimator = AngeliEstimator(dz0, A, X, W_unfold)
# controller = DirectController(estimator, gain=0.3)
controller = PIDController(estimator, Ki=0.3, Kp=0.3, Kd=0.0)



xs = []
x_hats = []
dxs = []
fwhms = []

dx = np.zeros(ndof)
for i in tqdm(range(100)):
    # rtp = rng.uniform(-np.pi/2, np.pi/2)
    rtp = 0.0
    # x = state_sim.next_x()
    x = state_sim.next_x(dx)
    y, y_noise = measure_sim.measurement(x, rtp)
    x_hat, dx = controller.control(y_noise, rtp, fields)
    xs.append(x)
    x_hats.append(x_hat)
    dxs.append(dx)
    fwhms.append(metrics.spot_size(x))
    # fwhms.append(0.0)
xs = np.array(xs)
x_hats = np.array(x_hats)
dxs = np.array(dxs)
fwhms = np.array(fwhms)



import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(16, 10))
axes = axes.ravel()
for i in range(ndof):
    axes[i].plot(xs[:, i], c='b', lw=2, alpha=0.5, label='true')
    axes[i].plot(x_hats[:, i], c='r', ls=':', label='est')
    axes[i].plot(dxs[:, i], c='g', ls='--', label='control')
axes[10].plot(fwhms)
axes[0].legend()

axes[0].set_title("M2 dz")
axes[1].set_title("M2 dx")
axes[2].set_title("M2 dy")
axes[3].set_title("M2 Rx")
axes[4].set_title("M2 Ry")

axes[5].set_title("cam dz")
axes[6].set_title("cam dx")
axes[7].set_title("cam dy")
axes[8].set_title("cam Rx")
axes[9].set_title("cam Ry")

axes[10].set_title("FWHM")

fig.tight_layout()
plt.show()