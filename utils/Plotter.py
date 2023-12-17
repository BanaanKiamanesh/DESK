import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    """
    A class used to plot the states of a Pendulum

    ...

    Attributes
    ----------
    t : array
        time vector
    sol : array
        solution from the simulation
    config : dict
        a dictionary containing the system properties

    Methods
    -------
    plot_results()
        Plots the states of the system in subplots.
    """

    def __init__(self, t, sol):
        self.t = t
        self.sol = sol

    def PlotResults(self):
        """Plots the states of the system in subplots."""
        fig, axs = plt.subplots(4)
        fig.suptitle('Rotary Inverted Pendulum',
                     fontweight='bold', fontsize=18)

        labels = [r'$\bf {\theta(t)}$', r'$\bf {\dot{\theta}}(t)$',
                  r'$\bf{\alpha(t)}$', r'$\bf{\dot{\alpha}(t)}$']
        y_labels = [r'${\bf \theta}$', r'$\bf {\dot{\theta}}$',
                    r'${\bf \alpha}$', r'${\bf \dot{\alpha}}$']

        for i, label in enumerate(labels):
            axs[i].plot(self.t, 
                        self.sol[:, i], 
                        color=['b', 'g', 'r', 'm'][i], 
                        label=label)
            axs[i].set_xlabel('t', fontweight='bold')
            axs[i].set_ylabel(y_labels[i], fontsize=14)
            axs[i].grid(which='both', 
                        color='#999999',
                        linestyle='-', 
                        alpha=0.2)
            axs[i].minorticks_on()
            axs[i].legend(loc='best')

        plt.show()
