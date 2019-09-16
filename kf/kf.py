import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt 
import scipy.stats as stats 
import control as ctrl

def getMeasurement(x, R):
    return x + np.random.normal(0, np.sqrt(R))

if __name__=="__main__":
    m = 100.0 #kg
    b = 20.0 #Ns/m
    ts = 0.05 #s
    F = 50.0 #N

    A = np.array([[0, 1],[0, -b/m]])
    B = np.array([[0, 1/m]]).T
    C = np.array([[1, 0]])
    D = np.array([0])

    sys = ctrl.StateSpace(A, B, C, D)
    sysd = sys.sample(ts, method="zoh") 
    Ad = sysd.A 
    Bd = sysd.B
    Cd = sysd.C

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
    
    Q = np.diag([0.0001, 0.01])
    R = .001  # Measurement noise
    Sigma = np.eye(2)*.01 # Initial covariance
    mu = np.array([[0.0, 0.0]]).T
    x = np.array([[0.0, 0.0]]).T

    for i in range(t.size):
        x_est_hist.append(mu.item(0))
        v_est_hist.append(mu.item(1))
        x_cov_hist.append(Sigma[0,0])
        v_cov_hist.append(Sigma[1,1])
        x_hist.append(x.item(0))  # have separate variable for truth.
        v_hist.append(x.item(1))
        err = x - mu
        x_err_hist.append(err.item(0))
        v_err_hist.append(err.item(1))

        if t[i] < 5:
            u = F
        elif t[i] < 25:
            u = 0
        elif t[i] < 30:
            u = -F
        else:
            u = 0

        #Prediction step
        mu_bar = Ad @  mu + Bd * u
        x = Ad @ x + Bd * u + Q @ np.random.normal(size=(2,1))
        Sigma_bar = Ad @ Sigma @ Ad.T + Q

        zt = getMeasurement(x.item(0), R)

        #Calculate Kalman gain
        Kt = Sigma_bar @ Cd.T @ npl.inv(Cd @ Sigma_bar @ Cd.T + R)
        K_hist.append(Kt)

        #Measurement Update
        mu = mu_bar + Kt @ (zt - Cd @ mu_bar)
        Sigma = (np.eye(2) - Kt @ Cd) @ Sigma_bar
    
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
    K_hist = np.array(K_hist).reshape(1000,2)
    plt.plot(t, K_hist[:,0])
    plt.plot(t, K_hist[:,1])
    plt.xlabel("Time (s)")
    plt.ylabel("Kalman Gain")
    plt.title("Kalman Gain vs Time")

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