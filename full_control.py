import numpy as np

from state import DiagLoopStateSim
from measurement import MeasSim
import rubin
from control import LstSqEstimator, DirectController, PIDController, AngeliEstimator
from metric import Metrics
from tqdm import tqdm


rng = np.random.default_rng(57722)
fields = [(-1.15, -1.15), (-1.15, 1.15), (1.15, -1.15), (1.15, 1.15)]
dof_idx = tuple(range(50))
ndof = len(dof_idx)
band = "r"
fiducial, wavelength = rubin.get_fiducial("r")
dz0 = rubin.get_dz0("r")
A = rubin.get_sensitivity_dz("r", dof_idx)
metrics = Metrics(dz0, A)

# Xdiag = np.array([10**2, 50**2, 50**2, 50**2, 50**2]*2)
Xdiag = np.array([5**2, 25**2, 25**2, 25**2, 25**2]*2+[0.1**2]*40)
X = np.diag(Xdiag)
x0 = rng.normal(scale=np.sqrt(Xdiag)*10, size=(ndof))
state_sim = DiagLoopStateSim(x0, Xdiag, time_scale=10, rng=rng)

W = np.diag([20.0**2]*(4*29))
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
for i in tqdm(range(40)):
    rtp = rng.uniform(-np.pi/2, np.pi/2)
    # x = state_sim.next_x()
    x = state_sim.next_x(dx)
    y, y_noise = measure_sim.measurement(x, rtp)
    x_hat, dx = controller.control(y_noise, rtp, fields)
    xs.append(x)
    x_hats.append(x_hat)
    dxs.append(dx)
    fwhms.append(metrics.spot_size(x))
xs = np.array(xs)
x_hats = np.array(x_hats)
dxs = np.array(dxs)
fwhms = np.array(fwhms)



names = ["M2 "+s for s in ["dz", "dx", "dy", "Rx", "Ry"]]
names += ["cam "+s for s in ["dz", "dx", "dy", "Rx", "Ry"]]
names += ["M1M3 mode "+str(i) for i in range(20)]
names += ["M2 mode "+str(i) for i in range(20)]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}
import matplotlib
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt


fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(16, 10))
axes = axes.ravel()
for i in range(ndof):
    axes[i].plot(xs[:, i], c='b', lw=2, alpha=0.5, label='true')
    axes[i].plot(x_hats[:, i], c='r', ls=':', label='est')
    axes[i].plot(dxs[:, i], c='g', ls='--', label='control')
    axes[i].set_title(names[i])
axes[0].legend()
axes[50].plot(fwhms)

axes[50].set_title("FWHM")

fig.tight_layout()
plt.show()