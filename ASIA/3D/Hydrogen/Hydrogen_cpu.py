import numpy as np

from scipy.sparse.linalg import eigs, eigsh
from scipy import sparse

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from skimage import measure

import time

N = 120
X, Y, Z = np.mgrid[-25:25:N*1j,-25:25:N*1j,-25:25:N*1j]
dx = np.diff(X[:,0,0])[0]

hbar = 1.055e-34
a = 5.29e-11
m = 9.11e-31
J_to_ev = 6.248e18
conversion = hbar**2 * a /(m * dx**2) * J_to_ev

def get_potential(x,y,z):

    return -dx**2 / np.sqrt(x**2 + y**2 + z**2 + 1e-10)

V = get_potential(X,Y,Z)

diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)
T = -1/2 * sparse.kronsum(sparse.kronsum(D,D),D)
U = sparse.diags(V.reshape(N**3), (0))
H = T + U

start = time.time()
eigenvalues, eigenvectors = eigsh(H, k=2, which='SM')
end = time.time() - start

print("Duration: %.2f s" % end)

k = np.linspace(0, len(eigenvalues)-1, len(eigenvalues))

fig, ax = plt.subplots()
ax.scatter(k, eigenvalues * conversion)

plt.show()

def get_e(n):

    return eigenvectors.T[n].reshape(N,N,N)

verts, faces, _, _ = measure.marching_cubes(get_e(1)**2, 1e-6, spacing=(0.1,0.1,0.1))
intensity = np.linalg.norm(verts, axis=1)

fig = go.Figure(data = [go.Mesh3d(
    xverts = verts[:,0], y = verts[:,1], z = verts[:,2],
    i = faces[:,0], j = faces[:,1], k = faces[:,2],
    intensity = intensity,
    colorscale = "Agsunset",
    opacity = 0.5)]
)

fig.update_layout(
    scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible = False),
        zaxis = dict(visible = False),
        bgcolor='rgb(0,0,0)'),
    margin = dict(l = 0, r = 0, b = 0, t = 0)
)
