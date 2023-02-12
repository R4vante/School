import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import scipy
# plt.rcParams["text.usetex"] = True


N = 1000
dx = 1/N

# potential length is L so if from -1 to 1 -> potential goes from -L/2 to L/2
x = np.linspace(0,1,N)

psi_an = np.sqrt(2)*np.sin(1*np.pi * x)/(scipy.linalg.norm(np.sqrt(2)*np.sin(1*np.pi * x)))

def get_zero_potential(x):
    return x*0

def get_finite(x):
    return 100*((x<=0.25)+(x>0.75)).astype(float)

def get_eigen(dx, V):

    main_diag = 1/dx**2 + V[1:-1]
    off_diag = -1/(2*dx**2) * np.ones(len(main_diag)-1)
    eigenvalues, eigenvectors = eigh_tridiagonal(main_diag, off_diag)

    return eigenvalues, eigenvectors.T


if __name__ == "__main__":

    V = get_zero_potential(x)
    
    E, psi = get_eigen(dx, V)
    n = np.arange(0, len(E[0:4]), 1)

    for i in range(0, len(psi)):
        dot = psi[i]@psi[i]
        
        if dot < 0.99:
            print("psi_%i is not orthogonal" %i)
            break

    # Validation
    fig, ax = plt.subplots(1,1, figsize=(12,12))
    ax.plot(x[1:-1], psi[0]**2, linewidth=3, label = "numerically")
    ax.plot(x, np.abs(psi_an)**2, 'r--', label="analytically")
    ax.grid(True)
    ax.set_xlabel(r"$\frac{x}{L}$", fontsize=15)
    ax.set_ylabel(r"$\left|\psi\right|^2$", fontsize=15)
    ax.legend()
    # Plot 1: Infinite square well

    fig2, ax2 = plt.subplots(1,2, figsize=(12, 12))
    for i in range(0,4):
        ax2[0].plot(x[1:-1], psi[i]**2)


    ax2[1].bar(n, E[0:4]/E[0])
    ax2[0].grid(True)
    ax2[0].set_xlabel(r"$\frac{x}{L}$", fontsize=15)
    ax2[0].set_ylabel(r"$\left|\psi\right|^2$", fontsize=15)
    ax2[1].set_xlabel(r"$n$", fontsize=15)
    ax2[1].set_ylabel(r"$\frac{mL^2E}{\hbar^2}$", fontsize=15)

    V = get_finite(x)
    E2, psi2 = get_eigen(dx, V)
    n2 = np.arange(0, len(E2[0:4]), 1)

    fig3, ax3 = plt.subplots(1,2, figsize=(12,12))
    

    for i in range(0,4):
        ax3[0].plot(x[1:-1], psi2[i]**2)

    ax3[1].bar(n2, E2[0:4]/E2[0])


    ax3[0].grid(True)
    ax3[0].set_xlabel(r"$\frac{x}{L}$", fontsize=15)
    ax3[0].set_ylabel(r"$\left|\psi\right|^2$", fontsize=15)
    ax3[1].set_xlabel(r"$n$", fontsize=15)
    ax3[1].set_ylabel(r"$\frac{mL^2E}{\hbar^2}$", fontsize=15)

    plt.show()
    

