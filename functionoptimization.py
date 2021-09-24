from matplotlib import pyplot as plt
from skimage import data, color
from scipy import ndimage as ndi
from scipy import optimize
import numpy as np

def astronaut_shift_error(shift, image):
    corrected = ndi.shift(image, (0, shift))
    return mse(astronaut, corrected)


def mse(arr1, arr2):
    return np.mean((arr1-arr2)**2)

astronaut = color.rgb2gray(data.astronaut())
shifted = ndi.shift(astronaut, (0, 5))
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(astronaut)
axes[0].set_title('original')
plt.show()
axes[1].imshow(shifted)
axes[1].set_title('shifted')
plt.show()
ncol = astronaut.shape[1]
shifts = np.linspace(-0.9*ncol, 0.9*ncol, 181)
mse_costs = []

for shift in shifts:
    shifted_back = ndi.shift(shifted, (0, shift))
    mse_costs.append(mse(astronaut, shifted_back))

fig, ax = plt.subplots()
ax.plot(shifts, mse_costs)
ax.set_xlabel('shift')
ax.set_ylabel('mse')

res = optimize.minimize(astronaut_shift_error, 0, args=(shifted,), method='Powell')
print(f'The optimal shift for correction is {res.x}')
ncol = astronaut.shape[1]
shifts = np.linspace(-0.9*ncol, 0.9*ncol, 181)
mse_costs = []
for shift in shifts:
    shifted1 = ndi.shift(astronaut, (0, shift))
    mse_costs.append(mse(astronaut, shifted1))

fig, ax = plt.subplots()
ax.plot(shifts, mse_costs)
ax.set_xlabel('shift')
ax.set_ylabel('mse')
shifted2 = ndi.shift(astronaut, (0, -340))
res = optimize.minimize(astronaut_shift_error,0, args=(shifted2,), method='Powell')
