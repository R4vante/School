import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from scipy.sparse.linalg import eigs, eigsh
from scipy import sparse

import torch
from torch import lobpcg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import time


class S_GPU:

    def __init__(self, N):

        self.N = N
        self.X, self.Y = np.mgrid[0:1:N*1j, 0:1:N*1j]

    def get_potential(self):

        self.V = np.zeros((len(self.X), len(self.Y)))
    
        return self.V
    
    def calc_psi(self):
        
        diag = np.ones([self.N])
        diags = np.array([diag, -2*diag, diag])

        self.D = sparse.spdiags(diags, np.array([-1, 0, 1]), self.N, self.N)

        self.T = -1/2 * sparse.kronsum(self.D, self.D)
        self.U = sparse.diags(self.get_potential().reshape(self.N**2), (0))

        self.H = self.T + self.U

        self.H = self.H.tocoo()
        self.H = torch.sparse_coo_tensor(indices=torch.tensor(np.array([self.H.row, self.H.col])), values=torch.tensor(self.H.data), size=self.H.shape).to(device)

        self.eigenvalues, self.eigenvectors = lobpcg(self.H, k=10, largest=False)

        return self.eigenvalues, self.eigenvectors

    def get_e(self, i):

        return self.eigenvectors.T[i].reshape(self.N,self.N).cpu()

    def plot(self, n):

        fig = go.Figure(data=[go.Surface(x=self.X, y=self.Y, z= self.get_e(n)**2)])

        fig.update_layout(autosize=False,
                                width = 1000, height=1000)

        fig.show()



    def test(self):

        st = time.time()

        self.calc_psi()

        elapsed_1 = time.time() -st

        print("-------------------------------------\n")
        print(f"\nN = {self.N}")
        print(f"Execution time using GPU:  {elapsed_1} seconds.\n")

def main():

    N = 150

    S = S_GPU(N)
    S.calc_psi()
    # S.test()


    # N = 300

    # S = S_GPU(N)
    # S.test()

    S.plot(0)


if __name__ == "__main__":

    main()