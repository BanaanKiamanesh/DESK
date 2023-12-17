import numpy as np
from scipy.integrate import ode
from utils.Plotter import Plotter
from tqdm import tqdm


class InvPendulum:
    """
    A class to represent Rotary Inverted Pendulum Dynamical System
    ...

    Attributes
    ----------


    Methods
    -------
        ode_system(t, x)
            Defines the ODE to be solved.
        simulate(x0, t, solver)
            Simulates the system for given initial conditions and time points.
            Available solvers: 'dopri5', 'vode', 'zvode', 'lsoda', 'dop853'
    """

    def __init__(self, config=None):
        default_config = {'g': 9.81,
                          'Rm': 2.6,
                          'Eta_g': 0.9,
                          'Eta_m': 0.69,
                          'kt': 7.68e-3,
                          'km': 7.68e-3,
                          'Kg': 70,
                          'Lp': 0.337,
                          'Mp': 0.127,
                          'Bp': 0,
                          'Br': 0,
                          'Lr': 0.216,
                          'Mr': 0.257,
                          }

        if config is None:
            config = default_config
        for key in default_config.keys():
            try:
                setattr(self, key, config[key])
            except KeyError:
                raise ValueError(f"Invalid config. Missing key: {key}")

        self.config = config

        # Define Inertia Values
        self.Jp = self.Mp * self.Lp**2 / 3
        self.Jr = self.Mr * self.Lr**2 / 3

    def ODEFunction(self, t, State):
        """Defines the ODE to be solved."""

        Theta, dTheta, Alpha, dAlpha = State
        dState = np.zeros(4)

        # State Derivative Calculation
        # Motor Torque Calculation
        # Let's Get the Input into Zero For Now
        Vm = 0
        Tau = self.Eta_g * self.Kg * self.Eta_m * self.kt \
                         * (Vm - self.Kg*self.km*dTheta) / self.Rm

        # A Matrix Filling
        A = np.zeros((2, 2))

        A[0, 0] = self.Mp * self.Lr**2 + 0.25 * self.Mp \
            * self.Lp**2 * (1 - np.cos(Alpha)**2) \
            + self.Jr
        A[0, 1] = -0.5 * self.Mp * self.Lp * self.Lr * np.cos(Alpha)
        A[1, 0] = -0.5 * self.Mp * self.Lp * self.Lr * np.cos(Alpha)
        A[1, 1] = self.Jp + 0.25 * self.Mp * self.Lp**2

        # B Matrix Filling
        B = np.zeros(2)
        B[0] = Tau - self.Br*dTheta - (0.5 * self.Mp * self.Lp**2 * np.sin(Alpha)
                                       * np.cos(Alpha)) * dTheta * dAlpha \
                                    - (0.5 * self.Mp * self.Lp *
                                       self.Lr * np.sin(Alpha)) * dAlpha**2

        B[1] = -self.Bp*dAlpha + (0.25*self.Mp * self.Lp**2
                                  * np.cos(Alpha) * np.sin(Alpha)) * dTheta**2 \
            + (0.5*self.Mp * self.Lp * self.g * np.sin(Alpha))

        # Solve for dX2 and dX4
        dState[1], dState[3] = np.linalg.solve(A, B)

        dState[0] = dTheta
        dState[2] = dAlpha

        return dState

    def Simulate(self, x0=np.zeros(4), t=np.linspace(0, 10), solver='dopri5'):
        """Simulates the system for given initial conditions and time points."""

        if solver not in ['dopri5', 'vode', 'zvode', 'lsoda', 'dop853']:
            raise ValueError(
                "Invalid solver. Available options are: 'dopri5', 'vode', 'zvode', 'lsoda', 'dop853'")

        # Solver Properties Selection
        r = ode(self.ODEFunction).set_integrator(solver)
        r.set_initial_value(x0, t[0])

        # Memoery Allocation and Initial Condition Set
        x = np.zeros((len(t), len(x0)))
        x[0, :] = x0

        for i, _t in tqdm(enumerate(t[1:]), total=len(t[1:]), desc="Simulating in Progress!"):
            r.integrate(_t)
            if r.successful():
                x[i+1, :] = r.y
            else:
                print(f"Integration failed at t={_t}")
                print("Return code:", r.get_return_code())
                break

        return x, t


if __name__ == "__main__":
    Pendy = InvPendulum()

    # System Simulation
    sol, t = Pendy.Simulate(x0=np.random.rand(4),
                            t=np.linspace(0, 15, 5000),
                            solver='dopri5')
    print(sol[-1, :])   # Print Final States

    print("Simulation Runs Beautifully :)" if np.allclose(
        sol[-1, :][2], np.pi, 2) else "Something is Wrong!")

    # Plotter Object Creation
    Plt = Plotter(t, sol)

    # Plot Results
    Plt.PlotResults()
