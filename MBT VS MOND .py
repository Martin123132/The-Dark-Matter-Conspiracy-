# %%
import numpy as np
import matplotlib.pyplot as plt

# === Load observed data ===
from astropy.io import fits

filename = "hlsp_frontier_model_abell2744_merten_v1_kappa.fits"
hdul = fits.open(filename)
data = hdul[0].data
hdul.close()

# === Prepare grid ===
shape = data.shape
y, x = np.indices(shape)

# === Define 3 optimized Gaussians ===
def gaussian2d(amplitude, x0, y0, sigma_x, sigma_y, theta, x, y):
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return amplitude * np.exp( - (a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2))

# Parameters from optimizer
params = [
    (5.3646, 93.45, 87.22, 6.88, 8.75, 0.8171),
    (0.3067, 82.22, 102.06, 5.26, 5.63, 0.4816),
    (-4.6994, 94.09, 86.66, 8.20, 6.76, -0.7646)
]

# === Build MBT field ===
mbt_field = sum(gaussian2d(*p, x, y) for p in params)

# === Compute RMSE ===
error = data - mbt_field
rmse = np.sqrt(np.nanmean(error**2))
print("✅ Optimized RMSE:", rmse)

# === Plot observed vs model ===
fig, axs = plt.subplots(1,3, figsize=(15,5))

im0 = axs[0].imshow(data, cmap='plasma')
axs[0].set_title("Observed Kappa Map")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(mbt_field, cmap='plasma')
axs[1].set_title("Optimized MBT Model")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(error, cmap='seismic', vmin=-0.3, vmax=0.3)
axs[2].set_title("Residual (Observed - Model)")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize

# === Load observed kappa FITS file ===
filename = "hlsp_frontier_model_abell2744_merten_v1_kappa.fits"  # Replace with your actual file path
hdul = fits.open(filename)
data = hdul[0].data
hdul.close()

# === Prepare X, Y grids ===
grid_y, grid_x = np.indices(data.shape)

# === Define MBT model: sum of rotated Gaussians ===
def mbt_model(grid_x, grid_y, flat_params):
    model = np.zeros_like(grid_x, dtype=float)
    num_gaussians = len(flat_params) // 6
    for i in range(num_gaussians):
        A, x0, y0, sx, sy, theta = flat_params[i*6:(i+1)*6]
        xp = (grid_x - x0)*np.cos(theta) + (grid_y - y0)*np.sin(theta)
        yp = -(grid_x - x0)*np.sin(theta) + (grid_y - y0)*np.cos(theta)
        exponent = -0.5 * ((xp/sx)**2 + (yp/sy)**2)
        model += A * np.exp(exponent)
    return model

# === Define RMSE loss function ===
def rmse_loss(flat_params):
    model = mbt_model(grid_x, grid_y, flat_params)
    error = data - model
    return np.sqrt(np.nanmean(error**2))

# === Initial guess for 5 Gaussians ===
initial_params = [
    0.8, 90, 90, 8, 8, 0,
    0.3, 80, 95, 5, 5, 0,
    0.2, 100, 85, 6, 4, np.pi/4,
    0.1, 85, 100, 7, 6, np.pi/6,
    0.15, 95, 80, 5, 7, -np.pi/6
]

# === Bounds for each parameter: (amplitude, x0, y0, sx, sy, theta) ===
bounds = []
for _ in range(5):
    bounds.extend([
        (0, 2),               # amplitude
        (0, data.shape[1]),   # x0
        (0, data.shape[0]),   # y0
        (1, 20),              # sigma_x
        (1, 20),              # sigma_y
        (-np.pi, np.pi)       # theta
    ])

# === Run optimization ===
result = minimize(rmse_loss, initial_params, bounds=bounds, method='L-BFGS-B')

# === Extract optimized parameters and compute final model ===
opt_params = result.x
final_model = mbt_model(grid_x, grid_y, opt_params)
final_rmse = rmse_loss(opt_params)

# === Print final RMSE and parameters ===
print(f"Final RMSE: {final_rmse:.5f}")
print("\nOptimized Parameters (5 Gaussians):")
for i in range(5):
    A, x0, y0, sx, sy, theta = opt_params[i*6:(i+1)*6]
    print(f"Gaussian {i+1}: A={A:.4f}, x0={x0:.2f}, y0={y0:.2f}, sx={sx:.2f}, sy={sy:.2f}, theta={theta:.4f}")

# === Plot observed, model, and residuals ===
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

im0 = axs[0].imshow(data, cmap='plasma')
axs[0].set_title("Observed Kappa Map")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(final_model, cmap='plasma')
axs[1].set_title("Optimized MBT Model")
plt.colorbar(im1, ax=axs[1])

residual = data - final_model
im2 = axs[2].imshow(residual, cmap='seismic', vmin=-0.5, vmax=0.5)
axs[2].set_title("Residual Map")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize

# === Load observed kappa FITS file ===
filename = "hlsp_frontier_model_abell2744_merten_v1_kappa.fits"  # Replace with your actual file path
hdul = fits.open(filename)
data = hdul[0].data
hdul.close()

# === Mask low-signal regions (e.g., kappa < 0.01) ===
mask = data > 0.01
grid_y, grid_x = np.indices(data.shape)

# === MBT model: sum of rotated Gaussians ===
def mbt_model(grid_x, grid_y, flat_params):
    model = np.zeros_like(grid_x, dtype=float)
    for i in range(len(flat_params) // 6):
        A, x0, y0, sx, sy, theta = flat_params[i*6:(i+1)*6]
        xp = (grid_x - x0)*np.cos(theta) + (grid_y - y0)*np.sin(theta)
        yp = -(grid_x - x0)*np.sin(theta) + (grid_y - y0)*np.cos(theta)
        model += A * np.exp(-0.5 * ((xp/sx)**2 + (yp/sy)**2))
    return model

# === RMSE loss function with masking ===
def rmse_loss(flat_params):
    model = mbt_model(grid_x, grid_y, flat_params)
    return np.sqrt(np.nanmean((data[mask] - model[mask])**2))

# === Initial guess for 6 Gaussians ===
initial_params = [
    0.8, 90, 90, 8, 8, 0,
    0.3, 80, 95, 5, 5, 0,
    0.2, 100, 85, 6, 4, np.pi/4,
    0.1, 85, 100, 7, 6, np.pi/6,
    0.15, 95, 80, 5, 7, -np.pi/6,
    0.12, 88, 92, 6, 6, np.pi/8
]

# === Bounds for each parameter ===
bounds = []
for _ in range(6):
    bounds += [
        (0, 2),               # amplitude
        (0, data.shape[1]),   # x0
        (0, data.shape[0]),   # y0
        (1, 20),              # sigma_x
        (1, 20),              # sigma_y
        (-np.pi, np.pi)       # theta
    ]

# === Run optimization ===
result = minimize(rmse_loss, initial_params, bounds=bounds, method='L-BFGS-B')
opt_params = result.x
final_model = mbt_model(grid_x, grid_y, opt_params)
final_rmse = rmse_loss(opt_params)

# === Print final RMSE and parameters ===
print(f"Final RMSE: {final_rmse:.5f}")
for i in range(6):
    A, x0, y0, sx, sy, theta = opt_params[i*6:(i+1)*6]
    print(f"Gaussian {i+1}: A={A:.4f}, x0={x0:.2f}, y0={y0:.2f}, sx={sx:.2f}, sy={sy:.2f}, theta={theta:.4f}")

# === Plot observed, model, and residuals ===
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(data, cmap='plasma')
axs[0].set_title("Observed Kappa Map")
plt.colorbar(axs[0].images[0], ax=axs[0])

axs[1].imshow(final_model, cmap='plasma')
axs[1].set_title("Optimized MBT Model (6 Gaussians)")
plt.colorbar(axs[1].images[0], ax=axs[1])

residual = data - final_model
axs[2].imshow(residual, cmap='seismic', vmin=-0.5, vmax=0.5)
axs[2].set_title("Residual Map")
plt.colorbar(axs[2].images[0], ax=axs[2])

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import differential_evolution

# === Load observed kappa FITS file ===
filename = "hlsp_frontier_model_abell2744_merten_v1_kappa.fits"  # Replace with your actual file path
hdul = fits.open(filename)
data = hdul[0].data
hdul.close()

# === Mask low-signal regions (e.g., kappa < 0.01) ===
mask = data > 0.01
grid_y, grid_x = np.indices(data.shape)

# === MBT model: sum of rotated Gaussians ===
def mbt_model(grid_x, grid_y, flat_params):
    model = np.zeros_like(grid_x, dtype=float)
    for i in range(len(flat_params) // 6):
        A, x0, y0, sx, sy, theta = flat_params[i*6:(i+1)*6]
        xp = (grid_x - x0)*np.cos(theta) + (grid_y - y0)*np.sin(theta)
        yp = -(grid_x - x0)*np.sin(theta) + (grid_y - y0)*np.cos(theta)
        model += A * np.exp(-0.5 * ((xp/sx)**2 + (yp/sy)**2))
    return model

# === RMSE loss function with masking ===
def rmse_loss(flat_params):
    model = mbt_model(grid_x, grid_y, flat_params)
    return np.sqrt(np.nanmean((data[mask] - model[mask])**2))

# === Bounds for 6 Gaussians ===
bounds = []
for _ in range(6):
    bounds += [
        (0, 2),               # amplitude
        (0, data.shape[1]),   # x0
        (0, data.shape[0]),   # y0
        (1, 20),              # sigma_x
        (1, 20),              # sigma_y
        (-np.pi, np.pi)       # theta
    ]

# === Run global optimization ===
result = differential_evolution(rmse_loss, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=1e-6)
opt_params = result.x
final_model = mbt_model(grid_x, grid_y, opt_params)
final_rmse = rmse_loss(opt_params)

# === Print final RMSE and parameters ===
print(f"Final RMSE: {final_rmse:.5f}")
for i in range(6):
    A, x0, y0, sx, sy, theta = opt_params[i*6:(i+1)*6]
    print(f"Gaussian {i+1}: A={A:.4f}, x0={x0:.2f}, y0={y0:.2f}, sx={sx:.2f}, sy={sy:.2f}, theta={theta:.4f}")

# === Plot observed, model, and residuals ===
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(data, cmap='plasma')
axs[0].set_title("Observed Kappa Map")
plt.colorbar(axs[0].images[0], ax=axs[0])

axs[1].imshow(final_model, cmap='plasma')
axs[1].set_title("Optimized MBT Model (6 Gaussians)")
plt.colorbar(axs[1].images[0], ax=axs[1])

residual = data - final_model
axs[2].imshow(residual, cmap='seismic', vmin=-0.5, vmax=0.5)
axs[2].set_title("Residual Map")
plt.colorbar(axs[2].images[0], ax=axs[2])

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# === Load your FITS file ===
filename = "hlsp_frontier_model_abell2744_merten_v1_kappa.fits"  # Replace with your actual file path
hdul = fits.open(filename)
data = hdul[0].data
hdul.close()

# === Create coordinate grid ===
grid_y, grid_x = np.indices(data.shape)

# === Optimized parameters from global fit ===
opt_params = [
    0.3163, 96.18, 97.34, 20.00, 20.00, -0.0660,
    0.0000, 80.01, 95.02, 5.03, 5.02, 0.0000,
    0.0000, 100.60, 85.08, 6.35, 4.29, 0.6021,
    0.0000, 85.73, 101.24, 8.51, 7.34, -0.1965,
    0.0000, 94.94, 80.03, 5.08, 7.03, -0.5076,
    0.7793, 89.31, 89.89, 4.05, 2.55, 1.1722
]

# === Function to compute a single rotated Gaussian ===
def rotated_gaussian(grid_x, grid_y, A, x0, y0, sx, sy, theta):
    xp = (grid_x - x0)*np.cos(theta) + (grid_y - y0)*np.sin(theta)
    yp = -(grid_x - x0)*np.sin(theta) + (grid_y - y0)*np.cos(theta)
    return A * np.exp(-0.5 * ((xp/sx)**2 + (yp/sy)**2))

# === Compute each Gaussian ===
gaussians = []
for i in range(6):
    A, x0, y0, sx, sy, theta = opt_params[i*6:(i+1)*6]
    g = rotated_gaussian(grid_x, grid_y, A, x0, y0, sx, sy, theta)
    gaussians.append(g)

# === Sum of all Gaussians ===
mbt_model = sum(gaussians)

# === Residuals ===
residual = data - mbt_model

# === Plot each Gaussian ===
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
for i, ax in enumerate(axs.flat):
    im = ax.imshow(gaussians[i], cmap='plasma')
    ax.set_title(f"Gaussian {i+1}")
    plt.colorbar(im, ax=ax)
plt.suptitle("Individual Gaussian Components", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# === Plot observed, model, and residuals ===
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(data, cmap='plasma')
axs[0].set_title("Observed Kappa Map")
plt.colorbar(axs[0].images[0], ax=axs[0])

axs[1].imshow(mbt_model, cmap='plasma')
axs[1].set_title("Full MBT Model (6 Gaussians)")
plt.colorbar(axs[1].images[0], ax=axs[1])

axs[2].imshow(residual, cmap='seismic', vmin=-0.5, vmax=0.5)
axs[2].set_title("Residual Map (Observed - Model)")
plt.colorbar(axs[2].images[0], ax=axs[2])

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import differential_evolution

# === Load observed kappa FITS file ===
filename = "hlsp_frontier_model_abell2744_merten_v1_kappa.fits"  # Make sure this file is in your working directory
hdul = fits.open(filename)
data = hdul[0].data
hdul.close()

# === Mask low-signal regions (e.g., kappa < 0.01) ===
mask = data > 0.01
grid_y, grid_x = np.indices(data.shape)

# === MBT model: sum of rotated Gaussians ===
def mbt_model(grid_x, grid_y, flat_params):
    model = np.zeros_like(grid_x, dtype=float)
    for i in range(len(flat_params) // 6):
        A, x0, y0, sx, sy, theta = flat_params[i*6:(i+1)*6]
        xp = (grid_x - x0)*np.cos(theta) + (grid_y - y0)*np.sin(theta)
        yp = -(grid_x - x0)*np.sin(theta) + (grid_y - y0)*np.cos(theta)
        model += A * np.exp(-0.5 * ((xp/sx)**2 + (yp/sy)**2))
    return model

# === RMSE loss function with masking ===
def rmse_loss(flat_params):
    model = mbt_model(grid_x, grid_y, flat_params)
    return np.sqrt(np.nanmean((data[mask] - model[mask])**2))

# === Bounds for 7 Gaussians with extended sigma bounds ===
bounds = []
for _ in range(7):
    bounds += [
        (0, 2),                  # amplitude
        (0, data.shape[1]),      # x0
        (0, data.shape[0]),      # y0
        (1, 40),                 # sigma_x
        (1, 40),                 # sigma_y
        (-np.pi, np.pi)         # theta
    ]

# === Run global optimization ===
result = differential_evolution(rmse_loss, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=1e-6)
opt_params = result.x
final_model = mbt_model(grid_x, grid_y, opt_params)
final_rmse = rmse_loss(opt_params)

# === Print final RMSE and parameters ===
print(f"Final RMSE: {final_rmse:.5f}")
for i in range(7):
    A, x0, y0, sx, sy, theta = opt_params[i*6:(i+1)*6]
    print(f"Gaussian {i+1}: A={A:.4f}, x0={x0:.2f}, y0={y0:.2f}, sx={sx:.2f}, sy={sy:.2f}, theta={theta:.4f}")

# === Plot observed, model, and residuals ===
residual = data - final_model
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(data, cmap='plasma')
axs[0].set_title("Observed Kappa Map")
plt.colorbar(axs[0].images[0], ax=axs[0])

axs[1].imshow(final_model, cmap='plasma')
axs[1].set_title("Optimized MBT Model (7 Gaussians)")
plt.colorbar(axs[1].images[0], ax=axs[1])

axs[2].imshow(residual, cmap='seismic', vmin=-0.5, vmax=0.5)
axs[2].set_title("Residual Map")
plt.colorbar(axs[2].images[0], ax=axs[2])
plt.tight_layout()
plt.show()

# === Plot individual Gaussians ===
def rotated_gaussian(grid_x, grid_y, A, x0, y0, sx, sy, theta):
    xp = (grid_x - x0)*np.cos(theta) + (grid_y - y0)*np.sin(theta)
    yp = -(grid_x - x0)*np.sin(theta) + (grid_y - y0)*np.cos(theta)
    return A * np.exp(-0.5 * ((xp/sx)**2 + (yp/sy)**2))

gaussians = []
for i in range(7):
    A, x0, y0, sx, sy, theta = opt_params[i*6:(i+1)*6]
    g = rotated_gaussian(grid_x, grid_y, A, x0, y0, sx, sy, theta)
    gaussians.append(g)

fig, axs = plt.subplots(3, 3, figsize=(18, 12))
for i, ax in enumerate(axs.flat[:7]):
    im = ax.imshow(gaussians[i], cmap='plasma')
    ax.set_title(f"Gaussian {i+1}")
    plt.colorbar(im, ax=ax)
plt.suptitle("Individual Gaussian Components (7 Total)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


