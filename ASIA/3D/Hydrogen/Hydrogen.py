import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigs, eigsh
from scipy import sparse

import torch
from torch import lobpcg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import plotly.graph_objs as go

from skimage import measure

import time

N = 120
X, Y, Z = np.mgrid[-25:25:N*1j, -25:25:N*1j, -25:25:N*1j] 
dx = np.diff(X[:,0,0])[0]

## Define Potential
def get_potential(x, y, z):

    return -dx**2/np.sqrt(x**2 + y**2 + z**2 + 1e-10)

V = get_potential(X,Y,Z)

diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1,0,1]), N, N)
T = -1/2 * sparse.kronsum(sparse.kronsum(D,D),D)
U = sparse.diags(V.reshape(N**3), (0))
H = T + U

H = H.tocoo()
H = torch.sparse_coo_tensor(indices = torch.tensor(np.array([H.row, H.col])), values = torch.tensor(H.data), size = H.shape).to(device)

start = time.time()
eigenvalues, eigenvectors = lobpcg(H, k=2, largest=False)
end = time.time() - start
print("Duration: %.2f s" % end)

hbar = 1.055e-34
a = 5.29e-11
m = 9.11e-31
J_to_ev = 6.2428e18
conversion = hbar**2 * a**2 / (m * dx**2) * J_to_ev

k = np.linspace(0, len(eigenvalues)-1, len(eigenvalues))

fig, ax = plt.subplots()
ax.scatter(k, eigenvalues.cpu() * conversion)

plt.show()

def get_e(n):

    return eigenvectors.T[n].reshape(N,N,N).cpu().numpy()

verts, faces, _, _ = measure.marching_cubes(get_e(1)**2, 1e-6, spacing=(0.1, 0.1, 0.1))
intensity = np.linalg.norm(verts, axis=1)

fig = go.Figure(data = [go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                                  i = faces[:,0], j=faces[:,1], k=faces[:,2],
                                  intensity = intensity,
                                  colorscale="Agsunset",
                                  opacity=0.5)])

fig.update_layout(scene=dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
    bgcolor='rgb(0,0,0)'),
    margin=dict(l=0, r=0, b=0, t=0)
)

fig.show()
