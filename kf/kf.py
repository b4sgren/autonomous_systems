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
    est_err_hist = []
    K_hist = []
    x_hist = []
    x_est_hist = []
    v_hist = []
    v_est_hist = []
    err_cov_hist = []
    
    #Figure handles
    # x_fig, x_ax = plt.subplots(2,1)
    # v_fig, v_zx = plt.subplots(2,1)
    # K_fig, K_ax = plt.subplots(1,1)
    # err_fig, err_ax = plt.subplots(1,1)
    
    Q = np.array([[.0001, 0], [0, 0.01]]) # Process noise
    R = .001  # Measurement noise
    Sigma = np.eye(2)*.01 # Initial covariance
    mu = np.array([[0.0, 0.0]]).T

    for i in range(t.size):
        x_est_hist.append(mu.item(0))
        v_est_hist.append(mu.item(1))
        err_cov_hist.append(Sigma)

        zt = getMeasurement(mu.item(0), R)
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
        Sigma_bar = Ad @ Sigma @ Ad.T + Q
        x_hist.append(mu_bar.item(0)) #Not sure that this is truth. But it is linear so maybe.
        v_hist.append(mu_bar.item(1))

        #Calculate Kalman gain
        Kt = Sigma_bar @ Cd.T @ npl.inv(Cd @ Sigma_bar @ Cd.T + R)
        K_hist.append(Kt)

        #Measurement Update
        mu = mu_bar + Kt @ (zt - Cd @ mu_bar)
        Sigma = (np.eye(2) - Kt @ Cd) @ Sigma_bar
    
    plt.figure(1)
    plt.plot(t, x_est_hist, 'b')
    plt.plot(t, x_hist, 'r')

    plt.plot(t, v_est_hist, 'g')
    plt.plot(t, v_hist, 'y')
    
    plt.figure(2)
    K_hist = np.array(K_hist).reshape(1000,2)
    plt.plot(t, K_hist[:,0])
    plt.plot(t, K_hist[:,1])

    plt.show()

    # mean = 8.0  # Ex plotting a normal distribution
    # stddev = 5.0

    # x = np.linspace(0, 16, 100)
    # y = stats.norm.pdf(x, mean, stddev)

    # plt.plot(x, y)
    # plt.show()