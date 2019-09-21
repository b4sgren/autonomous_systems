import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import scipy.stats as stats
import control as ctrl
import scipy.io as sio
from IPython.core.debugger import Pdb

def getMeasurement(x, R):
    return x + np.random.normal(0, np.sqrt(R))

if __name__=="__main__":
    m = 100.0 #kg
    b = 20.0 #Ns/m
    ts = 0.05 #s
    F = 50.0 #N
    read_file = True

    A = np.array([[0, 1],[0, -b/m]])
    B = np.array([[0, 1/m]]).T
    C = np.array([[1, 0]])
    D = np.array([0])

    sys = ctrl.StateSpace(A, B, C, D)
    sysd = sys.sample(ts, method="zoh")
    Ad = np.array(sysd.A)
    Bd = np.array(sysd.B)
    Cd = np.array(sysd.C)

    data = sio.loadmat('hw1_soln_data.mat')

    #Histories to plot
    x_err_hist = []
    v_err_hist = []
    K_hist = []
    x_hist = []
    x_est_hist = []
    v_hist = []
    v_est_hist = []
    x_cov_hist = []
    v_cov_hist = []

    Q = data['R']
    temp = Q.item(0)
    Q[0,0] = Q[1,1]
    Q[1,1] = temp
    R = data['Q'].item(0)
    mu = data['mu0']
    temp = mu.item(0)
    mu[0,0] = mu[1,0]
    mu[1,0] = temp
    u_l = data['u']
    t = data['t']
    xtr = data['xtr']
    vtr = data['vtr']
    z = data['z']
    Sigma = data['Sig0']
    temp = Sigma.item(0)
    Sigma[0,0] = Sigma[1,1]
    Sigma[1,1] = temp

    # Pdb().set_trace()
    for i in range(t.size):
        x_est_hist.append(mu.item(0))
        v_est_hist.append(mu.item(1))
        x_cov_hist.append(Sigma[0,0])
        v_cov_hist.append(Sigma[1,1])
        x_hist.append(xtr.item(i))
        v_hist.append(vtr.item(i))
        x = np.array([[xtr.item(i), vtr.item(i)]]).T
        err = x - mu
        x_err_hist.append(err.item(0))
        v_err_hist.append(err.item(1))

        u = u_l.item(i)

        #Prediction step
        mu_bar = Ad @  mu + Bd * u
        Sigma_bar = Ad @ Sigma @ Ad.T + Q

        zt = z.item(i)

        #Calculate Kalman gain
        Kt = Sigma_bar @ Cd.T @ npl.inv(Cd @ Sigma_bar @ Cd.T + R)
        K_hist.append(Kt)

        #Measurement Update
        mu = mu_bar + Kt @ (zt - Cd @ mu_bar)
        Sigma = (np.eye(2) - Kt @ Cd) @ Sigma_bar

    t = t.flatten()
    plt.figure(1)
    plt.plot(t, x_est_hist, 'b', label="Pos Est")
    plt.plot(t, x_hist, 'r', label="Pos Truth")

    plt.plot(t, v_est_hist, 'g', label="Vel Est")
    plt.plot(t, v_hist, 'y', label="Vel Truth")
    plt.legend()
    plt.ylabel("Estimate")
    plt.xlabel("Time (s)")
    plt.title("State Estimate vs Time")

    plt.figure(2)
    K_hist = np.array(K_hist).reshape(1001,2)
    plt.plot(t, K_hist[:,0], label="Position")
    plt.plot(t, K_hist[:,1], label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Kalman Gain")
    plt.title("Kalman Gain vs Time")
    plt.legend()

    plt.figure(3)
    x_cov_hist = np.sqrt(np.array(x_cov_hist)) * 2
    plt.plot(t, x_err_hist, 'b', label="Pos Error")
    plt.plot(t, x_cov_hist, 'r', label='2 sigma')
    plt.plot(t, -x_cov_hist, 'r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title("Position Error vs Time")

    plt.figure(4)
    v_cov_hist = np.sqrt(np.array(v_cov_hist)) * 2
    plt.plot(t, v_err_hist, 'b', label="Vel Error")
    plt.plot(t, v_cov_hist, 'r', label='2 sigma')
    plt.plot(t, -v_cov_hist, 'r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Error (m/s)")
    plt.title("Velocity Error vs Time")

    plt.show()

'''
Question 1: Yes the estimator does work as expected. The estimate tracks the truth pretty closely with most levels of noise (both process and measurement)
Question 2: Covariance increases after the prediction step and decreases after the measurement update
Question 3: Gains spike right at the beginning but after the first iteration they settle into steady state values almost immediately and stay there.
This is because the initial covariace is initally quite high compared with the covariance after a few iterations when the filter has "settled"
Increasing the measurement noise decreases the kalman gains while decreasing the measurement noise increases them.
'''
