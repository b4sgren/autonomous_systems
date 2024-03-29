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

    if read_file:
        data = sio.loadmat('hw1_soln_data.mat')
    else:
        t = np.arange(0, 50, ts)

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

    if read_file:
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
        Pdb().set_trace()
        temp = Sigma.item(0)
        Sigma[0,0] = Sigma[1,1]
        Sigma[1,1] = temp
        x = np.array([[0.0, 0.0]]).T
    else:
        Q = np.diag([0.0001, 0.01])  * 1.0
        R = .001 * 1.0  # Measurement noise
        # Sigma = np.eye(2) * 1 # Initial covariance
        Sigma = np.array([[1.0, 0.0], [0.0, 0.1]])
        mu = np.array([[-2.0, 2.0]]).T
        x = np.array([[0.0, 0.0]]).T

    # Pdb().set_trace()
    for i in range(t.size):
        x_est_hist.append(mu.item(0))
        v_est_hist.append(mu.item(1))
        x_cov_hist.append(Sigma[0,0])
        v_cov_hist.append(Sigma[1,1])
        if read_file:
            x_hist.append(xtr.item(i))
            v_hist.append(vtr.item(i))
        else:
            x_hist.append(x.item(0))  # have separate variable for truth.
            v_hist.append(x.item(1))
        err = x - mu
        x_err_hist.append(err.item(0))
        v_err_hist.append(err.item(1))
        x = np.array([[xtr.item(i), vtr.item(i)]]).T

        if read_file:
            u = u_l.item(i)
        elif t[i] < 5:
            u = F
        elif t[i] < 25:
            u = 0
        elif t[i] < 30:
            u = -F
        else:
            u = 0

        #Prediction step
        mu_bar = Ad @  mu + Bd * u
        if not read_file:
            x = Ad @ x + Bd * u + np.sqrt(Q) @ np.random.normal(size=(2,1))#np.random.multivariate_normal(np.zeros(2), Q).reshape((2,1))
        Sigma_bar = Ad @ Sigma @ Ad.T + Q

        if read_file:
            zt = z.item(i)
        else:
            zt = getMeasurement(x.item(0), R)

        #Calculate Kalman gain
        Kt = Sigma_bar @ Cd.T @ npl.inv(Cd @ Sigma_bar @ Cd.T + R)
        K_hist.append(Kt)

        #Measurement Update
        mu = mu_bar + Kt @ (zt - Cd @ mu_bar)
        Sigma = (np.eye(2) - Kt @ Cd) @ Sigma_bar

    if read_file:
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
    if read_file:
        K_hist = np.array(K_hist).reshape(1001,2)
    else:
        K_hist = np.array(K_hist).reshape(1000,2)
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
