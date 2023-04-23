import matplotlib
import numpy as np
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt 
import scipy as sp
from scipy.sparse import linalg as ln
from scipy import sparse as sparse
import matplotlib.animation as animation
from IPython.display import HTML
 

class Wave_packet:

    def __init__(self, N, dt, sigma0=5.0, k0=1.0, x0=-150.0, x_begin=-200.0,
                x_end=200.0, barrier_height=1.0, barrier_width=3.0):
        
        self.N = N
        self.dt = dt
        self.sigma0 = sigma0
        self.k0 = k0
        self.x0 = x0
        self.x_begin = x_begin
        self.x_end = x_end
        self.barrier_height = barrier_height
        self.barrier_width = barrier_width

        self.x, self.dx = np.linspace(self.x_begin, self.x_end, self.N, retstep=True)

        norm = (2.0 * np.pi * self.sigma0**2)**(-0.25)

        self.psi = norm * np.exp(-(self.x - self.x0)**2 / (4.0 * self.sigma0**2))*np.exp(1.0j * self.k0 * self.x)
        self.potential = self.barrier_height * ((self.x >= 0) * (self.x < self.barrier_width)).astype(float)
        main = 1/(self.dx**2) * np.ones(self.N) + self.potential
        off = -1/(2*self.dx**2) * np.ones(len(main)-1)
        H = sparse.diags([main, off, off], [0, 1, -1])

        implicit = (sparse.eye(self.N) - self.dt / 2.0j * H).tocsc()
        explicit = (sparse.eye(self.N) + self.dt / 2.0j * H).tocsc()

        self.evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()

    def evolve(self):
        self.psi = self.evolution_matrix.dot(self.psi)
        self.prob = abs(self.psi)**2

        norm = np.sum(self.prob)

        self.prob /= norm
        self.psi /= norm**(0.5)

        return self.prob


class Animator:
    
    def __init__(self, wavepacket):
        self.time = 0.0
        self.wavepacket = wavepacket
        self.fig, self.ax = plt.subplots()
        plt.plot(self.wavepacket.x, self.wavepacket.potential*0.1, color='r')

        self.line, = self.ax.plot(self.wavepacket.x, self.wavepacket.evolve())

    def time_step(self):

        while True:
            self.time += self.wavepacket.dt

            yield self.wavepacket.evolve()

    def update(self, data):
        self.line.set_ydata(data)
        return self.line,

    def animate(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames = self.time_step, interval=5, blit=False, save_count =500
        )
        # writer = animation.PillowWriter(fps=30)
        # self.ani.save('tunnel.gif', writer=writer)


def main():

    packet = Wave_packet(N=500, dt=0.5, barrier_width=1000, barrier_height = 0.5)
    animator = Animator(packet)
    animator.animate()
    plt.show()


if __name__ == "__main__":

    main()