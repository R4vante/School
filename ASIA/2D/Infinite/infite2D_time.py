import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, eigsh
from scipy import sparse
import time

import torch
from torch import lobpcg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class S_CPU:

    def __init__(self, N):

        self.N = N
        self.X, self.Y = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))

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
        
        self.eigenvalues, self.eigenvectors = eigsh(self.H, k=10, which='SM')

        return self.eigenvalues, self.eigenvectors


class S_GPU:

    def __init__(self, N):
        
        self.N = N
        self.X, self.Y = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))

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
        self. H = torch.sparse_coo_tensor(indices=torch.tensor(np.array([self.H.row, self.H.col])), values=torch.tensor(self.H.data), size=self.H.shape).to(device)

        self.eigenvalues, self.eigenvectors = lobpcg(self.H, k=10, largest=False)

        return self.eigenvalues, self.eigenvectors

def test(N):

    S_cpu = S_CPU(N)
    S_gpu = S_GPU(N)

    st = time.time()

    S_cpu.calc_psi()

    elapsed_1 = time.time() - st

    st = time.time()

    S_gpu.calc_psi()

    elapsed_2 = time.time() -st 


    print("-------------------------------------\n")
    print(f"\nN = {N}")
    print(f"Execution time using CPU:  {elapsed_1} seconds.\n")
    print(f"Execution time using GPU:  {elapsed_2} seconds.\n")

        




def main():


    N = 150

    test(N)

    N=300

    test(N)



if __name__=='__main__':

    main()