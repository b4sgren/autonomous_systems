import car_params as params 
from pomdp import POMDPPlanner
import matplotlib.pyplot as plt
import numpy as np
            
if __name__ == "__main__":
    planner = POMDPPlanner()

    planner.createPolicy()
    print("Created the policy")
    print("Alpha Vectors: ")
    print(planner.Y)

    V, P = planner.getMax()

    plt.figure(1)
    plt.plot(P,V)
    plt.xlim([0,1])
    plt.ylim([-100, 100])
    plt.show()