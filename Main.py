from utils.InvPendulum import InvPendulum
import numpy as np
from utils.Plotter import Plotter

Pendy = InvPendulum()
Sol, t = Pendy.Simulate(x0=np.random.rand(4), t=np.linspace(0, 20, 1000))

Plt = Plotter(t, Sol)
Plt.PlotResults()
print(f"System Steady State: {Sol[-1,]}")
