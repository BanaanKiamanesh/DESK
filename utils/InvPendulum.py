import numpy as np
from scipy.integrate import ode
from utils.Plotter import Plotter
from tqdm import tqdm


def Trajectory(t):
    X = 0
    dX = 0
    ddX = 0
    return X, dX, ddX


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
                          'Vmax': 6
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

        # self.Jp = 1
        # self.Jr = 1

    def Saturation(self, X):
        return X if np.abs(X) > 1 else np.sign(X)

    def SlidingMode(self, t, State, Traj=Trajectory):
        # Desired X and Derivatives of it
        Alphad, dAlphad, ddAlphad = Traj(t)
        Thetad, dThetad, ddThetad = 0, 0, 0

        # Disturbance
        OmegaTheta = 0
        OmegaAlpha = 0

        # Unpack the in Coming State
        Theta, dTheta, Alpha, dAlpha = State

        # Error Dynamics
        # Alpha
        AlphaErr = Alphad - Alpha
        dAlphaErr = dAlphad - dAlpha
        # Theta
        ThetaErr = Thetad - Theta
        dThetaErr = dThetad - dTheta

        # Sliding Mode Controller Coefs
        Lambda1 = 0
        Eta = 0
        Lambda2 = 0

        # Sliding Surfaces
        STheta = dThetaErr + Lambda1 * ThetaErr
        SAlpha = dAlphaErr + Lambda1 * AlphaErr

        # Dynamical Propeties Calculation

        # Motor Torque Calculation
        Tau = self.Eta_g * self.Kg * self.Eta_m * self.kt / self.Rm
        # F Matrix Filling
        F = np.zeros((2, 2))

        F[0, 0] = self.Mp * self.Lr**2 + 0.25 * self.Mp \
            * self.Lp**2 * (1 - np.cos(Alpha)**2) \
            + self.Jr
        F[0, 1] = -0.5 * self.Mp * self.Lp * self.Lr * np.cos(Alpha)
        F[1, 0] = -0.5 * self.Mp * self.Lp * self.Lr * np.cos(Alpha)
        F[1, 1] = self.Jp + 0.25 * self.Mp * self.Lp**2

        # B Matrix Filling
        G = np.zeros(3)
        G[0] = Tau - self.Br*dTheta - (0.5 * self.Mp * self.Lp**2 * np.sin(Alpha)
                                       * np.cos(Alpha)) * dTheta * dAlpha \
            - (0.5 * self.Mp * self.Lp * self.Lr * np.sin(Alpha)) * dAlpha**2

        G[1] = self.Eta_g * self.Kg * self.Eta_m * self.kt / self.Rm - self.Bp*dAlpha + (0.25*self.Mp * self.Lp**2
                                                                                         * np.cos(Alpha) * np.sin(Alpha)) * dTheta**2 \
            + (0.5*self.Mp * self.Lp * self.g * np.sin(Alpha))

        G[2] = -self.Kg*self.km*dTheta / self.Rm

        # D as the Deteminant of F
        D = np.linalg.det(F)
        print(D)

        # Control Signal Calculation
        dSTheta = -Eta * np.sign(STheta)
        dSAlpha = -Eta * np.sign(SAlpha)

        U1 = (((-(dSTheta + OmegaTheta + Lambda1 * dTheta)
              * D + F[0, 1] * G[2]) / F[1, 1]) - G[1]) / G[0]
        U2 = (((-(dSAlpha + OmegaAlpha - Lambda1 * dAlphaErr - ddAlphad)
              * D + F[0, 0] * G[2]) / F[1, 0]) - G[1]) / G[0]

        Vm = (U1 * Lambda2 + U2) / (1 + Lambda2)
        Vm = U2
        return Vm
    
    def SlidingModeNew(self, t, State, Traj=Trajectory):
        # Desired X and Derivatives of it
        Alphad, dAlphad, ddAlphad = Traj(t)
        Thetad, dThetad, ddThetad = 0, 0, 0

        # Disturbance
        OmegaTheta = 0
        OmegaAlpha = 0

        # Unpack the in Coming State
        Theta, dTheta, Alpha, dAlpha = State

        # Dynamical Propeties Calculation

        # Generalized Coordinate
        # q = [Theta, Alpha] , dq = [dTheta, dAlpha] , ddq = [ddTheta, ddAlpha]
        q = np.zeros((2,1))
        q[0, 0] = Theta
        q[1, 0] = Alpha
        
        dq = np.zeros((2,1))
        dq[0, 0] = dTheta
        dq[1, 0] = dAlpha

        # Generalized Coordinate Desireds
        q_d = np.zeros((2,1))
        q_d[0, 0] = Thetad
        q_d[1, 0] = Alphad
        
        dq_d = np.zeros((2,1))
        dq_d[0, 0] = dThetad
        dq_d[1, 0] = dAlphad

        ddq_d = np.zeros((2,1))
        ddq_d[0, 0] = ddThetad
        ddq_d[1, 0] = ddAlphad


        # Generalized Coordinate Errors
        e  = q_d  - q
        de = dq_d - dq

        # Sliding Mode Controller Coefs
        Lambda = 10
        Eta = 3

        # Sliding Surface
        S = de + Lambda * e

        # M Matrix Filling
        M = np.zeros((2, 2))

        M[0, 0] = self.Mp * self.Lr**2 + 0.25 * self.Mp * self.Lp**2 - 0.25 * self.Mp * self.Lp**2 * np.cos(Alpha)**2 + self.Jr
        M[0, 1] = -0.5 * self.Mp * self.Lp * self.Lr * np.cos(Alpha)
        M[1, 0] = -0.5 * self.Mp * self.Lp * self.Lr * np.cos(Alpha)
        M[1, 1] = self.Jp + 0.25 * self.Mp * self.Lp**2

        # C Matrix Filling
        C = np.zeros((2, 2))

        C[0, 0] = (0.5 * self.Mp * self.Lp**2 * np.sin(Alpha) * np.cos(Alpha)) * Alphad + self.Br
        C[0, 1] =  0.5 * self.Mp * self.Lp * self.Lr * np.sin(Alpha) * Alphad
        C[1, 0] = (-0.25 * self.Mp * self.Lp**2 * np.sin(Alpha) * np.cos(Alpha)) * Thetad
        C[1, 1] = self.Bp

        # G Vector Filling
        G = np.zeros((2, 1))

        G[0, 0] = 0
        G[1, 0] = -0.5 * self.Mp * self.Lp * self.g * np.sin(Alpha)

        # D as the Deteminant of F
        # D = np.linalg.det(M)
        # print(D)

        # Control Signal Calculation
        U = M * (ddq_d + Lambda * e + Eta * np.tanh(S)) + C * dq + G

        # Relationship Between Motor Torque and Voltage
        Vm = self.Rm / (self.Eta_g * self.Kg * self.Eta_m * self.kt) * U[0, 0] + self.Kg * self.km * dTheta 
        
        return Vm


    def ForcedSystemODE(self, t, State):
        # self.U = self.SlidingMode(t, State)
        self.U = self.SlidingModeNew(t, State)
        # self.U = 0
        return self.SystemODE(t, State, self.U)

    def SystemODE(self, t, State, Vm):
        """Defines the ODE to be solved.
            >> Note that Vm is the System Input.
        """

        # Vm = np.clip(Vm, -self.Vmax, self.Vmax)

        Theta, dTheta, Alpha, dAlpha = State
        dState = np.zeros(4)

        # State Derivative Calculation
        # Motor Torque Calculation
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

    def Simulate(self, x0=np.zeros(4), t=np.linspace(0, 10), solver='dopri5', controller='SlidingModeNew'):
        """Simulates the system for given initial conditions and time points and Controller."""

        AvailableSolvers = ['dopri5', 'vode', 'zvode', 'lsoda', 'dop853']
        if solver not in AvailableSolvers:
            raise ValueError(
                f"Invalid solver. Available options are:{AvailableSolvers}")

        # Controller Properties
        AvailableControllers = ['SlidingMode', 'SlidingModeNew']
        if controller not in AvailableControllers:
            raise ValueError(
                f"Invalid Controller. Available options are: {AvailableControllers}")

        # Solver Properties Selection
        Solver = ode(self.ForcedSystemODE).set_integrator(solver)
        Solver.set_initial_value(x0, t[0])

        # Memoery Allocation and Initial Condition Set
        x = np.zeros((len(t), len(x0)))
        x[0, :] = x0

        for i, _t in tqdm(enumerate(t[1:]), total=len(t[1:]), desc="Simulating in Progress!"):
            Solver.integrate(_t)
            if Solver.successful():
                x[i+1, :] = Solver.y
            else:
                print(f"Integration failed at t={_t}")
                print("Return code:", Solver.get_return_code())
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
