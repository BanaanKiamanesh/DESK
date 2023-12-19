from utils.InvPendulum import InvPendulum
import numpy as np
from utils.Plotter import Plotter

Pendy = InvPendulum()
InitCond = np.random.rand(4)
tSpan = np.linspace(0, 20, 1000)
Sol, t = Pendy.Simulate(InitCond, tSpan)

Plt = Plotter(t, Sol)
Plt.PlotResults()
print(f"System Steady State: {Sol[-1,]}")
 