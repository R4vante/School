import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
from scipy.linalg import eigh_tridiagonal

N = 1000
dx = 1/N
x = np.linspace(-1, 1, N+1)


def mL2V(x):
    return 100*((x<=-0.5) + (x > 0.5)).astype(float)

def CalcSchroedinger(x, dx):
    main_diag = 1/(dx**2) + mL2V(x)[1:-1]
    off_diag = -1/(2*dx**2) * np.ones(len(main_diag)-1)

    E, psi = eigh_tridiagonal(main_diag, off_diag)

    return E, psi.T


if __name__ == "__main__":
    
    E, psi = CalcSchroedinger(x, dx)

    print("\n", 100*"-", "\n")
    print("\nTest orthogonality:\n")
    print("dot-product psi1, psi2: ", psi[0]@psi[1], "\n")
    print("dot-product psi1, psi1: ", psi[0]@psi[0])
    print("\n", 100*"-", "\n")

    if psi[0]@psi[1] <= 0.01 and psi[0]@psi[0] > 0.99:
        print("\nVectors are othogonal.\n")
        print("\n", 100*"-", "\n")

    else:
        print("\nVectors are not othogonal\n")
        print("\n", 100*"-", "\n")



    fig, ax = plt.subplots(1,1)
    ax.grid(True)
    ax.set_xlabel(r"$\frac{x}{L}$", fontsize=15)
    ax.set_ylabel(r"$\left|\psi\right|^2$", fontsize=15)
    
    for i in range(0,3):
        ax.plot(x[1:-1], np.abs(psi[i])**2)
    
    plt.show()
