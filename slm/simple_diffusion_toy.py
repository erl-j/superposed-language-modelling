#%%
import torch
from matplotlib import pyplot as plt

V = 2
D = 2
L = 1

We = torch.randn(V, D)
Wu = torch.randn(D, V)

x = torch.randint(0, V, (L,))
p0 = torch.eye(D)[x]

z0 = p0@We



# plot z0 vector in cartesian coordinate system
# x goes from -2 to 2, y goes from -2 to 2
# use an arrow for z0
plot_scale = 3
plt.quiver(0, 0, z0[0,0], z0[0,1], angles='xy', scale_units='xy', scale=1, color='b')
# plot rows of We as well
for i in range(V):
    plt.quiver(0, 0, We[i,0], We[i,1], angles='xy', scale_units='xy', scale=1, color='r')
plt.xlim(-plot_scale, plot_scale)
plt.ylim(-plot_scale, plot_scale)
plt.xlabel('x')
plt.ylabel('y')
plt.title('z0 vector')
plt.grid(True)
plt.show()

# %%
