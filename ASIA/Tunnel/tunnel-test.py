import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from matplotlib.animation import FuncAnimation, PillowWriter


class Wave_packet:
# NOTE: Code voor het maken van een golfpacket geven in de opgave. Verder uitleg geven 
# over hoe de class opgeroepen moet worden.

    def __init__(self, N, x0, sigma0, x_begin, x_end, k0 = 0):
       
        self.N = N
        self.x0 = x0
        self.sigma0 = sigma0
        self.x_begin = x_begin
        self.x_end = x_end
        self.k0 = k0
        self.x, self.dx = np.linspace(self.x_begin, self.x_end, self.N, retstep=True)        

    def packet(self):
        self.psi0 = np.exp(-(self.x-self.x0)**2/(2*self.sigma0)**2) * np.exp(-1j * self.k0 * self.x)
        norm = np.sum(np.abs(self.psi0)**2*self.dx, axis=0)
        self.psi0 = self.psi0 / np.sqrt(norm)
        return self.psi0
#---------------------------
# NOTE: Alles hieronder niet meeleveren in de opgave
    def potential(self):
        return 1 * ((self.x >= -0.5) * (self.x <=0.5)).astype(float)
        # a = -1/(0.5 + 0.5)
        # k1 = (3+0.5 + a *(self.x + 0.5)) * ((self.x>=-0.5) * (self.x <= 0.5))
        # k2 = (1+1)* ((self.x>=self.x_begin) * (self.x < -0.5))
        # return k1+k2
        # 
    def get_e(self):

        self.V = self.potential()
        main = 1/self.dx**2 + self.V
        off = -1/(2*self.dx**2) * np.ones(len(main)-1)

        self.E, self.psi = eigh_tridiagonal(main, off)

        self.psi = self.psi.T

        norm = np.sum(np.abs(self.psi)**2*self.dx, axis=0)
        self.psi = self.psi/np.sqrt(np.abs(norm))

        return self.E, self.psi


    def get_t(self, t):

        E, psi = self.get_e()
        psi0 = self.packet()

        self.cn = np.zeros_like(psi[0], dtype=complex)

        for j in range(0, self.N-1):

            self.cn[j] = np.sum(np.conj(psi[j]) * psi0 * self.dx)

        self.psi_t = psi.T@(self.cn*np.exp(-1j*E*t))
        norm = np.sum(np.abs(self.psi_t)**2 * self.dx)
        self.psi_t = self.psi_t / np.sqrt(norm)
        return self.psi_t


def main():
   
    packet = Wave_packet(1000, -20, 5, -150, 150, k0=-1)
    t = np.linspace(0, 25, 1000)

    psi0 = packet.packet()

    # NOTE: code voor de animatie niet meegeven, wel uitleggen

    fig  = plt.figure(figsize = (20,12))
    ax = plt.axes(xlim=(-50, 50))
    ax.plot(packet.x, np.max(np.abs(psi0)**2) * packet.potential()/np.max(packet.potential()), 'r--')

    ln, = ax.plot([],[])

    def animate(i):

        ln.set_data(packet.x, np.abs(packet.get_t(i))**2)
        return ln,

    # ani = FuncAnimation(fig, animate, frames = 100, interval=50, blit=False)
    # ani.save('test.mp4', fps=30, dpi=100)

    psi_l = packet.get_t(25)[packet.x <=0]
    psi_r = packet.get_t(25)[packet.x > 0]
    E, psi = packet.get_e()

    T = np.sum(np.abs(psi_r)**2 * packet.dx)/np.sum(np.abs(packet.get_t(25)**2) * packet.dx)
    R = np.sum(np.abs(psi_l)**2 * packet.dx)/np.sum(np.abs(packet.get_t(25)**2) * packet.dx)

    if E < np.max(packet.potential()):
        k1 = 1 + np.max(packet.potential())/(4*E * (np.max(packet.potential()) - E))
        k2 = 2 * np.sqrt(2*(np.max(packet.potential()) - E))

    T1 = k1 * np.sinh(k2)**2

    Tt = T1**(-1)

    print(T, Tt)


    plt.show()


if __name__ == "__main__":
#
    main()
